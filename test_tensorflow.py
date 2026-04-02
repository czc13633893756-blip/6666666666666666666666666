
"""
TensorFlow 快速测试脚本
运行此脚本验证TensorFlow是否能正常工作
"""

import sys
import tensorflow as tf

print("=" * 50)
print("TensorFlow 快速测试")
print("=" * 50)

print(f"Python 版本: {sys.version[:20]}...")
print(f"Python 路径: {sys.executable}")
print(f"TensorFlow 版本: {tf.__version__}")

# 检查GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ 找到 {len(gpus)} 个GPU设备:")
    for gpu in gpus:
        print(f"   - {gpu}")
else:
    print("ℹ️  未找到GPU设备，将使用CPU")

# 运行简单计算测试
print("\n运行计算测试...")
try:
    # 创建两个张量
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

    # 矩阵乘法
    c = tf.matmul(a, b)

    print(f"✅ 测试成功!")
    print(f"矩阵 a: \n{a.numpy()}")
    print(f"矩阵 b: \n{b.numpy()}")
    print(f"矩阵乘积 a×b: \n{c.numpy()}")

except Exception as e:
    print(f"❌ 测试失败: {e}")

print("=" * 50)
