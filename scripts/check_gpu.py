import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 메모리 설정 완료")
    except RuntimeError as e:
        print("GPU 메모리 설정 실패:", e)
else:
    print("사용 가능한 GPU가 없습니다.")

# CUDA 환경 변수 확인
print("\nCUDA 환경 변수:")
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH')) 