import spectral.io.envi as envi
import spectral
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from scipy.ndimage import label, find_objects, center_of_mass, binary_fill_holes, binary_opening, \
    binary_closing, gaussian_filter1d, binary_dilation, distance_transform_edt
from scipy.signal import find_peaks
import sys
import os
import logging
from skimage.filters import threshold_otsu
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 设置非交互式后端
matplotlib.use('Agg')

# ==========================================
# 1. 全局配置
# ==========================================
spectral.settings.envi_support_nonlowercase_params = True

# 路径配置
INPUT_FOLDER = Path(r"F:\1\2025-2炭疽侵染数据\7.2-36H")
OUTPUT_FOLDER = INPUT_FOLDER / "ROI_Verified_Results_RatioDiff"
OUTPUT_FOLDER.mkdir(exist_ok=True)
DEBUG_FOLDER = OUTPUT_FOLDER / "Debug_Details_RatioDiff"
DEBUG_FOLDER.mkdir(exist_ok=True)

# 结构先验参数
EXPECTED_LEAVES = 6
MIN_LEAF_PIXELS = 12000  # 最小有效叶片面积
PROJECTION_SIGMA = 25  # 投影曲线平滑程度 (根据图片高度调整，2000px高建议10-20)

# 比值+差值分割方法参数
RATIO_NUM_BAND_NM = 545.0  # 比值分子波段
RATIO_DEN_BAND_NM = 500.0  # 比值分母波段
DIFF_BAND1_NM = 545.0  # 差值波段1
DIFF_BAND2_NM = 672.0  # 差值波段2
RATIO_THRESHOLD = 0.022  # 比值阈值
DIFF_THRESHOLD = 0.022  # 差值阈值
USE_CALIBRATION = False  # 是否使用白板/黑板校正
USE_OTSU_FALLBACK = True  # 是否在阈值分割失败时回退到Otsu
WHITE_REF_PATH = None  # 白板参考立方体路径
BLACK_REF_PATH = None  # 黑板参考立方体路径
CALIBRATION_LOWER_PERCENTILE = 1  # 鲁棒归一化下限百分位
CALIBRATION_UPPER_PERCENTILE = 99  # 鲁棒归一化上限百分位

# 形态学参数
MORPH_CLOSING_KERNEL1 = (3, 41)  # 首次闭操作核（垂直方向）
MORPH_CLOSING_KERNEL2 = (7, 7)  # 二次闭操作核
MORPH_OPENING_KERNEL = (3, 3)  # 开操作核
FILL_HOLES = True  # 是否填充孔洞

# 可视化参数
INSET_PIXELS = 10  # ROI内缩像素数
OVERLAY_ALPHA = 0.6  # 叠加层透明度
CONTOUR_COLOR = 'red'  # 内缩轮廓颜色
CONTOUR_LINEWIDTH = 1.0  # 内缩轮廓线宽

ENVI_DTYPE = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_FOLDER / "segmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. 核心处理类
# ==========================================
class RatioDiffMethodLeafProcessor:
    def __init__(self, folder_path, white_ref_path=None, black_ref_path=None):
        self.folder_path = folder_path
        self.hdr_files = list(self.folder_path.glob("*.hdr"))
        if not self.hdr_files:
            logger.warning(f"在 {self.folder_path} 未找到 .hdr 文件。")
        self.wavelengths = np.linspace(400, 1000, 557)

        # 校正数据
        self.white_ref = None
        self.black_ref = None
        if white_ref_path and Path(white_ref_path).exists():
            self.white_ref = self.load_reference(white_ref_path)
        if black_ref_path and Path(black_ref_path).exists():
            self.black_ref = self.load_reference(black_ref_path)

        # 保存全局配置为实例属性
        self.EXPECTED_LEAVES = EXPECTED_LEAVES
        self.MIN_LEAF_PIXELS = MIN_LEAF_PIXELS
        self.PROJECTION_SIGMA = PROJECTION_SIGMA
        self.RATIO_NUM_BAND_NM = RATIO_NUM_BAND_NM
        self.RATIO_DEN_BAND_NM = RATIO_DEN_BAND_NM
        self.DIFF_BAND1_NM = DIFF_BAND1_NM
        self.DIFF_BAND2_NM = DIFF_BAND2_NM
        self.RATIO_THRESHOLD = RATIO_THRESHOLD
        self.DIFF_THRESHOLD = DIFF_THRESHOLD
        self.USE_CALIBRATION = USE_CALIBRATION
        self.USE_OTSU_FALLBACK = USE_OTSU_FALLBACK
        self.CALIBRATION_LOWER_PERCENTILE = CALIBRATION_LOWER_PERCENTILE
        self.CALIBRATION_UPPER_PERCENTILE = CALIBRATION_UPPER_PERCENTILE
        self.INSET_PIXELS = INSET_PIXELS
        self.OVERLAY_ALPHA = OVERLAY_ALPHA
        self.CONTOUR_COLOR = CONTOUR_COLOR
        self.CONTOUR_LINEWIDTH = CONTOUR_LINEWIDTH

        logger.info(f"初始化完成，找到 {len(self.hdr_files)} 个头文件。")
        logger.info(f"分割方法: 比值+差值法")
        logger.info(f"比值条件: {RATIO_NUM_BAND_NM}nm/{RATIO_DEN_BAND_NM}nm > {RATIO_THRESHOLD}")
        logger.info(f"差值条件: {DIFF_BAND1_NM}nm-{DIFF_BAND2_NM}nm > {DIFF_THRESHOLD}")
        logger.info(f"使用校正: {USE_CALIBRATION}, Otsu回退: {USE_OTSU_FALLBACK}")
        logger.info(f"ROI内缩像素: {INSET_PIXELS}, 叠加透明度: {OVERLAY_ALPHA}")

    def load_reference(self, ref_path):
        """加载白板/黑板参考数据"""
        try:
            ref_path = Path(ref_path)
            data_path = ref_path.with_suffix('')
            if not data_path.exists():
                for ext in ['.raw', '.img', '.dat', '.bin']:
                    if ref_path.with_suffix(ext).exists():
                        data_path = ref_path.with_suffix(ext)
                        break

            data, header = self.manual_read(ref_path, data_path)
            if data is not None:
                logger.info(f"成功加载参考数据: {ref_path.name}")
                return data
        except Exception as e:
            logger.warning(f"加载参考数据失败 {ref_path}: {e}")
        return None

    def find_nearest_band(self, target_nm):
        """找到最接近目标波长的波段索引"""
        return (np.abs(self.wavelengths - target_nm)).argmin()

    def manual_read(self, hdr_path, data_path):
        """手动读取ENVI数据"""
        try:
            header = envi.read_envi_header(str(hdr_path))
            lines = int(header.get('lines', 0))
            samples = int(header.get('samples', 0))
            bands = int(header.get('bands', 0))
            dtype_code = int(header.get('data type', 4))
            interleave = header.get('interleave', 'bsq').lower().strip()
            header_offset = int(header.get('header offset', 0))

            np_dtype = ENVI_DTYPE.get(dtype_code, np.float32)
            item_size = np.dtype(np_dtype).itemsize
            expected_bytes = lines * samples * bands * item_size
            file_size = os.path.getsize(data_path)

            if file_size < expected_bytes:
                logger.error(f"文件不完整: {data_path}")
                return None, header

            raw_flat = np.memmap(data_path, dtype=np_dtype, mode='r',
                                 offset=header_offset, shape=(lines * samples * bands,))

            if interleave == 'bsq':
                data = raw_flat.reshape((bands, lines, samples)).transpose((1, 2, 0))
            elif interleave == 'bil':
                data = raw_flat.reshape((lines, bands, samples)).transpose((0, 2, 1))
            else:
                data = raw_flat.reshape((lines, samples, bands))

            return data, header
        except Exception as e:
            logger.error(f"读取失败 {hdr_path}: {e}")
            return None, None

    def apply_reflectance_calibration(self, data, band_idx):
        """
        应用反射率校正: R = (I - B) / (W - B)
        其中: I=样本数据, B=黑板, W=白板
        """
        if self.white_ref is None or self.black_ref is None:
            logger.warning("缺少白板或黑板数据，跳过校正")
            return None

        try:
            # 提取对应波段的参考数据
            I = data[:, :, band_idx].astype(np.float32)
            B = self.black_ref[:, :, band_idx].astype(np.float32)
            W = self.white_ref[:, :, band_idx].astype(np.float32)

            # 避免除零
            denominator = W - B
            denominator[denominator == 0] = 1.0

            # 计算反射率
            reflectance = (I - B) / denominator

            # 裁剪到合理范围
            reflectance = np.clip(reflectance, 0.0, 1.0)

            logger.info(f"应用反射率校正: 波段索引 {band_idx}")
            return reflectance
        except Exception as e:
            logger.error(f"反射率校正失败: {e}")
            return None

    def robust_normalization(self, band_data):
        """
        鲁棒归一化: 使用百分位数裁剪后映射到[0,1]
        """
        data_flat = band_data.flatten()

        # 计算百分位数
        lower = np.percentile(data_flat, self.CALIBRATION_LOWER_PERCENTILE)
        upper = np.percentile(data_flat, self.CALIBRATION_UPPER_PERCENTILE)

        # 裁剪和归一化
        clipped = np.clip(band_data, lower, upper)
        normalized = (clipped - lower) / (upper - lower + 1e-10)

        return normalized

    def get_band_data(self, data, band_nm, file_id=""):
        """获取指定波段的数据，应用校正或归一化"""
        band_idx = self.find_nearest_band(band_nm)
        band_data = data[:, :, band_idx].astype(np.float32)

        band_data_norm = None

        if self.USE_CALIBRATION and self.white_ref is not None and self.black_ref is not None:
            # 尝试反射率校正
            band_data_norm = self.apply_reflectance_calibration(data, band_idx)
            if band_data_norm is None:
                logger.warning(f"{file_id}: 波段 {band_nm}nm 反射率校正失败，回退到鲁棒归一化")
                band_data_norm = self.robust_normalization(band_data)
        else:
            # 鲁棒归一化
            band_data_norm = self.robust_normalization(band_data)

        return band_data_norm, band_idx

    def ratio_diff_segmentation(self, data, file_id=""):
        """
        比值+差值分割方法实现
        返回: (binary_mask, ratio, diff, thresholds_used)
        """
        # 1. 获取所需波段的数据
        band545_norm, idx_545 = self.get_band_data(data, self.RATIO_NUM_BAND_NM, file_id)
        band500_norm, idx_500 = self.get_band_data(data, self.RATIO_DEN_BAND_NM, file_id)
        band672_norm, idx_672 = self.get_band_data(data, self.DIFF_BAND2_NM, file_id)

        # 2. 计算比值和差值
        # 避免除零
        eps = 1e-10

        # 比值: 545nm / 500nm
        ratio = band545_norm / (band500_norm + eps)

        # 差值: 545nm - 672nm
        diff = band545_norm - band672_norm

        # 3. 阈值分割
        mask1 = ratio > self.RATIO_THRESHOLD
        mask2 = diff > self.DIFF_THRESHOLD

        # 两个条件同时满足
        initial_mask = mask1 & mask2

        thresholds_used = (self.RATIO_THRESHOLD, self.DIFF_THRESHOLD)

        # 检查分割结果
        foreground_pixels = np.sum(initial_mask)

        # 回退策略：如果分割结果不理想，尝试分别使用Otsu阈值
        if foreground_pixels < self.MIN_LEAF_PIXELS and self.USE_OTSU_FALLBACK:
            # 对比值使用Otsu阈值
            otsu_threshold_ratio = threshold_otsu(ratio)
            mask1 = ratio > otsu_threshold_ratio

            # 对差值使用Otsu阈值
            otsu_threshold_diff = threshold_otsu(diff)
            mask2 = diff > otsu_threshold_diff

            initial_mask = mask1 & mask2
            thresholds_used = (float(otsu_threshold_ratio), float(otsu_threshold_diff))
            logger.info(f"{file_id}: 比值+差值分割失败，回退到Otsu阈值: 比值={otsu_threshold_ratio:.6f}, 差值={otsu_threshold_diff:.6f}")

        logger.info(
            f"{file_id}: 初始分割前景像素: {foreground_pixels}, 使用阈值: 比值={thresholds_used[0]:.6f}, 差值={thresholds_used[1]:.6f}")

        # 4. 形态学后处理
        mask = initial_mask.astype(np.uint8)

        # binary_closing
        mask = binary_closing(mask, structure=np.ones(MORPH_CLOSING_KERNEL1))
        mask = binary_closing(mask, structure=np.ones(MORPH_CLOSING_KERNEL2))

        # fill_holes
        if FILL_HOLES:
            mask = binary_fill_holes(mask)

        # opening
        mask = binary_opening(mask, structure=np.ones(MORPH_OPENING_KERNEL))

        # 5. 连通域过滤
        labeled_mask, num_features = label(mask)
        if num_features > 0:
            # 计算每个连通域的面积
            areas = np.bincount(labeled_mask.ravel())
            areas[0] = 0  # 忽略背景

            # 过滤小区域
            large_enough = [i for i in range(1, num_features + 1)
                            if areas[i] >= self.MIN_LEAF_PIXELS]

            if len(large_enough) > 0:
                # 按面积排序，保留最大的 N 个区域
                if len(large_enough) > self.EXPECTED_LEAVES + 2:
                    # 只保留面积最大的 EXPECTED_LEAVES + 2 个
                    large_enough.sort(key=lambda x: areas[x], reverse=True)
                    large_enough = large_enough[:self.EXPECTED_LEAVES + 2]

                # 创建新的mask
                filtered_mask = np.zeros_like(mask, dtype=bool)
                for label_idx in large_enough:
                    filtered_mask[labeled_mask == label_idx] = True

                mask = filtered_mask
            else:
                logger.warning(f"{file_id}: 没有找到足够大的连通域")
                mask = np.zeros_like(mask, dtype=bool)

        return mask.astype(bool), ratio, diff, thresholds_used

    def filter_components(self, mask, ndvi, min_area, min_mean_ndvi, keep_top):
        """
        全局连通域过滤：
        - 只保留面积 >= min_area 且平均NDVI >= min_mean_ndvi 的连通域
        - 最多保留 keep_top 个最大的连通域
        """
        lab, n = label(mask)
        if n == 0:
            return mask

        areas = np.bincount(lab.ravel())
        areas[0] = 0

        mean_nd = np.zeros(n + 1, dtype=np.float32)
        for i in range(1, n + 1):
            if areas[i] == 0:
                continue
            m = (lab == i)
            mean_nd[i] = float(ndvi[m].mean())

        good = (areas >= min_area) & (mean_nd >= min_mean_ndvi)
        idx = np.where(good)[0]
        if idx.size == 0:
            return np.zeros_like(mask, dtype=bool)

        # 只保留面积最大的 keep_top 个
        if idx.size > keep_top:
            idx = idx[np.argsort(areas[idx])[-keep_top:]]
            good2 = np.zeros_like(good, dtype=bool)
            good2[idx] = True
            good = good2

        return good[lab]

    def get_projection_strips(self, mask):
        """
        更稳的投影切条（用 proj_mask）
        """
        H, W = mask.shape

        # 投影专用 mask：轻微 opening 抑制毛刺/雪花点
        proj_mask = binary_opening(mask, structure=np.ones((7, 7)))
        row_sum = np.sum(proj_mask, axis=1)

        smooth_curve = gaussian_filter1d(row_sum, sigma=self.PROJECTION_SIGMA)

        # 峰查找参数建议更稳一点
        dist = max(20, H // 15)
        prom = max(1.0, smooth_curve.max() * 0.10)

        peaks, props = find_peaks(smooth_curve, distance=dist, prominence=prom)

        # 峰不足：补峰（按均分窗口找局部最大）
        if len(peaks) < self.EXPECTED_LEAVES:
            centers = np.linspace(H * 0.10, H * 0.90, self.EXPECTED_LEAVES).astype(int)
            win = max(50, H // 20)
            extra = []
            for c in centers:
                a = max(0, c - win)
                b = min(H, c + win)
                if b > a:
                    yy = np.argmax(smooth_curve[a:b]) + a
                    extra.append(yy)
            peaks = np.unique(np.concatenate([peaks, np.array(extra)]))

        # 峰过多：取最高的 EXPECTED_LEAVES 个
        if len(peaks) > self.EXPECTED_LEAVES:
            ph = smooth_curve[peaks]
            top = np.argsort(ph)[-self.EXPECTED_LEAVES:]
            peaks = np.sort(peaks[top])

        # valley 切割
        cut_lines = []
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i + 1]
                valley = np.argmin(smooth_curve[p1:p2]) + p1
                cut_lines.append(valley)

        boundaries = [0] + cut_lines + [H]
        return boundaries, peaks, smooth_curve

    def create_inset_mask(self, mask: np.ndarray, inset_px: int = 10) -> np.ndarray:
        """
        将二值mask向内缩进 inset_px 像素。
        使用距离变换：dist>inset_px 即保留离边界超过 inset_px 的内部区域。
        如果内缩后没有像素，则返回原mask
        """
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return mask

        # 计算欧氏距离变换
        dist = distance_transform_edt(mask)

        # 内缩：保留距离边界大于inset_px的区域
        inset = dist > float(inset_px)

        # 兜底：如果内缩后没有像素，返回原mask
        if inset.sum() == 0:
            logger.warning(f"内缩后无像素，返回原mask (面积: {mask.sum()} 像素)")
            return mask

        return inset

    def create_final_result_image(self, rgb, final_labels, temp_leaves, file_id):
        """
        创建最终结果图 (FinalResult)
        背景: RGB (660/550/460, 2-98% 拉伸)
        叠加: final_labels 以半透明不同颜色叠加
        编号: 在center_of_mass处标注Leaf_ID，红色圆形底 + 白字
        标题: Final Result (Found N)
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # 显示RGB背景
        ax.imshow(rgb)

        # 创建带alpha通道的彩色叠加层
        overlay = np.zeros((final_labels.shape[0], final_labels.shape[1], 4))

        # 使用tab20颜色映射
        cmap = plt.cm.get_cmap('tab20')

        # 为每个叶片创建彩色叠加
        for i, leaf in enumerate(temp_leaves):
            leaf_id = leaf['id']
            mask = (final_labels == leaf_id)

            if mask.any():
                # 为每个叶片分配颜色
                color = cmap(i % 20)
                overlay[mask] = [*color[:3], self.OVERLAY_ALPHA]  # RGB + alpha

        # 添加彩色叠加层
        ax.imshow(overlay)

        # 标注每个叶片的编号
        for leaf in temp_leaves:
            cy, cx = leaf['center']
            leaf_id = leaf['id']

            # 红色圆形底 + 白字
            ax.text(cx, cy, str(leaf_id),
                    color='white', fontweight='bold', fontsize=12,
                    ha='center', va='center',
                    bbox=dict(boxstyle="circle", facecolor="red", alpha=0.8, pad=1))

        # 设置标题
        ax.set_title(f"Final Result (Found {len(temp_leaves)})", fontsize=16, fontweight='bold')
        ax.axis('off')

        # 保存图片
        output_path = DEBUG_FOLDER / f"FinalResult_{file_id}.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"    -> 保存最终结果图: FinalResult_{file_id}.jpg")
        return output_path

    def create_inset_overlay_image(self, rgb, temp_leaves, file_id):
        """
        创建内缩mask叠加图 (InsetOverlay)
        背景: 同FinalResult的RGB
        叠加: 对每个叶片的global_mask做内缩10px，以半透明绿色叠加
        轮廓: 画出inset_mask的轮廓线
        编号: 在原始ROI的center位置标注（蓝底白字）
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # 显示RGB背景
        ax.imshow(rgb)

        # 创建内缩mask叠加层
        overlay = np.zeros((rgb.shape[0], rgb.shape[1], 4))

        # 为每个叶片创建内缩mask并叠加
        for i, leaf in enumerate(temp_leaves):
            leaf_id = leaf['id']
            original_mask = leaf['mask']

            # 生成内缩mask
            inset_mask = self.create_inset_mask(original_mask, self.INSET_PIXELS)
            leaf['inset_mask'] = inset_mask  # 保存到leaf字典中

            # 计算内缩后的像素数
            inset_pixel_count = inset_mask.sum()
            leaf['inset_pixel_count'] = inset_pixel_count

            # 以半透明绿色叠加内缩区域
            overlay[inset_mask] = [0.2, 0.8, 0.2, 0.6]  # 浅绿色，60%透明度

            # 绘制内缩mask的轮廓
            from skimage import measure

            # 找到轮廓
            contours = measure.find_contours(inset_mask.astype(np.uint8), 0.5)

            # 绘制所有轮廓
            for contour in contours:
                if len(contour) > 1:  # 确保轮廓有效
                    ax.plot(contour[:, 1], contour[:, 0],
                            linewidth=self.CONTOUR_LINEWIDTH,
                            color=self.CONTOUR_COLOR,
                            alpha=0.8)

        # 添加内缩mask叠加层
        ax.imshow(overlay)

        # 标注每个叶片的编号（使用原始中心位置，蓝底白字）
        for leaf in temp_leaves:
            cy, cx = leaf['center']
            leaf_id = leaf['id']

            # 蓝色圆形底 + 白字
            ax.text(cx, cy, str(leaf_id),
                    color='white', fontweight='bold', fontsize=12,
                    ha='center', va='center',
                    bbox=dict(boxstyle="circle", facecolor="blue", alpha=0.8, pad=1))

        # 设置标题
        ax.set_title(f"InsetMask Overlay (Found {len(temp_leaves)}, Inset={self.INSET_PIXELS}px)",
                     fontsize=16, fontweight='bold')
        ax.axis('off')

        # 保存图片
        output_path = DEBUG_FOLDER / f"InsetMaskOverlay_{file_id}.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"    -> 保存内缩mask叠加图: InsetMaskOverlay_{file_id}.jpg")
        return output_path

    def generate_rgb_background(self, full_data):
        """生成RGB背景图片 (660/550/460, 2-98% 拉伸)"""
        # 获取RGB波段索引
        r_idx = self.find_nearest_band(660)
        g_idx = self.find_nearest_band(550)
        b_idx = self.find_nearest_band(460)

        # 提取RGB通道
        rgb = full_data[:, :, [r_idx, g_idx, b_idx]].astype(np.float32)

        # 2-98%拉伸
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

        return rgb

    def process_single_file(self, hdr_path):
        file_id = hdr_path.stem

        # 查找数据文件
        data_path = hdr_path.with_suffix('')
        if not data_path.exists():
            for ext in ['.raw', '.img', '.dat', '.bin']:
                if hdr_path.with_suffix(ext).exists():
                    data_path = hdr_path.with_suffix(ext)
                    break
        if not data_path.exists():
            logger.warning(f"数据文件不存在: {hdr_path}")
            return []

        try:
            # 读取数据
            full_data, header = self.manual_read(hdr_path, data_path)
            if full_data is None:
                return []

            # 更新波长信息
            if header and 'wavelength' in header:
                wl = header['wavelength']
                if len(wl) == full_data.shape[2]:
                    self.wavelengths = np.array([float(w) for w in wl])

            full_data = np.nan_to_num(full_data)
            logger.info(f"处理文件: {file_id}, 形状: {full_data.shape}")

            # 1. 使用比值+差值分割方法获取Mask
            binary_mask, ratio, diff, thresholds_used = self.ratio_diff_segmentation(full_data, file_id)

            # 计算NDVI用于后续过滤
            nir_idx = self.find_nearest_band(800)
            red_idx = self.find_nearest_band(680)
            nir = full_data[:, :, nir_idx].astype(np.float32)
            red = full_data[:, :, red_idx].astype(np.float32)
            ndvi_map = (nir - red) / (nir + red + 1e-6)

            # 2. 投影分割核心逻辑
            boundaries, peaks, proj_curve = self.get_projection_strips(binary_mask)

            final_labels = np.zeros_like(binary_mask, dtype=np.int32)
            temp_leaves = []

            leaf_id_counter = 1

            # 遍历每个条带 (Strip)
            for i in range(len(boundaries) - 1):
                y_start, y_end = boundaries[i], boundaries[i + 1]

                # 提取条带 Mask
                strip_mask = binary_mask[y_start:y_end, :]

                # 连通域分析
                if np.sum(strip_mask) < self.MIN_LEAF_PIXELS:
                    # 空条带 (可能叶片缺失)
                    pass
                else:
                    # 标记连通域
                    labels, num_feat = label(strip_mask)
                    if num_feat > 0:
                        # 计算连通域大小
                        sizes = np.bincount(labels.ravel())
                        sizes[0] = 0  # 忽略背景

                        # 只考虑面积达到阈值的连通域
                        cands = [lb for lb in range(1, num_feat + 1)
                                 if sizes[lb] >= self.MIN_LEAF_PIXELS // 10]

                        if not cands:
                            continue

                        # 提取当前条带的NDVI
                        strip_ndvi = ndvi_map[y_start:y_end, :]

                        # 选"更像叶片"的：mean_ndvi最大
                        mean_ndvi = {lb: float(strip_ndvi[labels == lb].mean()) for lb in cands}
                        best = max(cands, key=lambda lb: mean_ndvi[lb])

                        # 可选：把与 best NDVI 接近且触碰的碎块也并进来（补阴影断裂）
                        best_mask = (labels == best)
                        touch = binary_dilation(best_mask, structure=np.ones((5, 25)))

                        keep = best_mask.copy()
                        for lb in cands:
                            if lb == best:
                                continue
                            if mean_ndvi[lb] > mean_ndvi[best] - 0.08 and np.any((labels == lb) & touch):
                                keep |= (labels == lb)

                        current_leaf_mask_local = keep

                        # 写入全局 Label Map
                        final_labels[y_start:y_end, :][current_leaf_mask_local] = leaf_id_counter

                        # 记录信息
                        global_mask = (final_labels == leaf_id_counter)
                        count = np.sum(global_mask)
                        if count > 0:  # 确保有像素
                            cy, cx = center_of_mass(global_mask)

                            temp_leaves.append({
                                'id': leaf_id_counter,
                                'mask': global_mask,
                                'center': (cy, cx),
                                'count': count
                            })

                # 无论是否找到叶片，ID 都要自增
                leaf_id_counter += 1
                if leaf_id_counter > self.EXPECTED_LEAVES:
                    break

            # 3. 生成RGB背景
            rgb = self.generate_rgb_background(full_data)

            # 4. 生成最终结果图
            self.create_final_result_image(rgb, final_labels, temp_leaves, file_id)

            # 5. 生成内缩mask叠加图
            self.create_inset_overlay_image(rgb, temp_leaves, file_id)

            # 6. 使用内缩mask提取光谱
            extracted = []
            for leaf in temp_leaves:
                # 获取内缩mask
                inset_mask = leaf.get('inset_mask', self.create_inset_mask(leaf['mask'], self.INSET_PIXELS))

                # 提取光谱
                if inset_mask.sum() > 0:
                    # 使用内缩mask提取平均光谱
                    mean_spec = np.mean(full_data[inset_mask], axis=0)

                    # 记录内缩前后的像素数
                    inset_pixel_count = inset_mask.sum()
                    original_pixel_count = leaf['count']

                    row = {
                        'FileName': file_id,
                        'Leaf_ID': leaf['id'],
                        'Pixel_Count': original_pixel_count,
                        'Inset_Pixel_Count': inset_pixel_count,
                        'Method': 'RatioDiffMethod'
                    }

                    # 添加光谱数据
                    for i, w in enumerate(self.wavelengths):
                        row[f"{w:.2f}nm"] = mean_spec[i]

                    extracted.append(row)
                else:
                    logger.warning(f"文件 {file_id} 叶片 {leaf['id']} 内缩后无像素，跳过光谱提取")

            logger.info(f"    -> 提取到 {len(extracted)} 片叶片的光谱数据")
            return extracted

        except Exception as e:
            logger.error(f"处理失败 {file_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def run(self):
        all_rows = []
        logger.info("开始处理... 策略: 比值+差值法")
        logger.info(f"期望叶片数: {self.EXPECTED_LEAVES} (竖直排列)")

        for hdr in self.hdr_files:
            logger.info(f"处理文件: {hdr.stem}")
            res = self.process_single_file(hdr)
            if res:
                all_rows.extend(res)
            else:
                logger.warning(f"跳过文件: {hdr.stem}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            # 排序：文件名 -> Leaf ID
            if 'Leaf_ID' in df.columns:
                df = df.sort_values(by=['FileName', 'Leaf_ID'])

            cols = ['FileName', 'Leaf_ID', 'Pixel_Count', 'Inset_Pixel_Count', 'Method'] + \
                   [c for c in df.columns if 'nm' in c]
            output_csv = OUTPUT_FOLDER / "Leaf_Specific_Spectra_RatioDiff.csv"
            df[cols].to_csv(output_csv, index=False)
            logger.info(f"处理完成。结果保存在: {output_csv}")
        else:
            logger.warning("未提取到任何数据。")


if __name__ == "__main__":
    # 使用示例
    processor = RatioDiffMethodLeafProcessor(
        INPUT_FOLDER,
        white_ref_path=WHITE_REF_PATH,  # 可选: 白板参考数据路径
        black_ref_path=BLACK_REF_PATH  # 可选: 黑板参考数据路径
    )
    processor.run()
