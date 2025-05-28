import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, MultiHeadAttention, Layer, Conv1D, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import pickle
import json


# GPU 설정
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print(f"GPU 사용 가능: {gpus[0]}")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed Precision 활성화됨")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

    # 기존 세션 정리 및 메모리 해제
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    # TensorFlow 최적화 설정
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "disable_model_pruning": False,
        "scoped_allocator_optimization": True,
        "pin_to_host_optimization": True,
        "implementation_selector": True,
        "auto_mixed_precision": True
    })

def save_model_and_scaler(model, scaler, model_index):
    """모델과 스케일러를 저장하는 함수"""
    try:
        # 모델 저장
        model_path = f'stock_prediction_model_{model_index}.keras'
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # NumPy 배열을 Python 기본 타입으로 변환하는 함수
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # 스케일러 상태 저장
        scaler_state = {
            'price_scaler': {
                'scale_': convert_to_serializable(scaler.price_scaler.scale_),
                'min_': convert_to_serializable(scaler.price_scaler.min_),
                'data_min_': convert_to_serializable(scaler.price_scaler.data_min_),
                'data_max_': convert_to_serializable(scaler.price_scaler.data_max_),
                'feature_names_in_': convert_to_serializable(scaler.price_scaler.feature_names_in_)
            },
            'feature_scaler': {
                'scale_': convert_to_serializable(scaler.feature_scaler.scale_),
                'min_': convert_to_serializable(scaler.feature_scaler.min_),
                'data_min_': convert_to_serializable(scaler.feature_scaler.data_min_),
                'data_max_': convert_to_serializable(scaler.feature_scaler.data_max_),
                'feature_names_in_': convert_to_serializable(scaler.feature_scaler.feature_names_in_)
            },
            'price_min': convert_to_serializable(scaler.price_min),
            'price_max': convert_to_serializable(scaler.price_max),
            'feature_names': convert_to_serializable(scaler.feature_names if hasattr(scaler, 'feature_names') else [])
        }
        
        scaler_path = f'stock_prediction_scaler_{model_index}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_state, f)
        print(f"Scaler state saved to {scaler_path}")
        
        # 모델 메타데이터 저장
        metadata = {
            'input_shape': convert_to_serializable(model.input_shape[1:]),
            'output_shape': convert_to_serializable(model.output_shape[1:]),
            'scaler_params': scaler_state
        }
        
        metadata_path = f'model_metadata_{model_index}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"Error saving model and scaler: {e}")
        raise

# 손실 함수
@tf.keras.utils.register_keras_serializable()
def enhanced_weighted_time_mse(y_true, y_pred):
    # 수치적 안정성을 위한 작은 값 추가
    epsilon = 1e-7
    
    # 입력 shape 확인 및 조정
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 5% 변동성 제한 적용
    y_pred = tf.clip_by_value(y_pred, -0.05, 0.05)
    
    # 시간별 가중치 조정 (장기 예측의 정확도 향상)
    time_weights = tf.constant([0.25, 0.2, 0.2, 0.2, 0.15], dtype=tf.float32)
    
    # 기본 MSE 손실
    base_loss = tf.reduce_mean(tf.square(y_true - y_pred) * time_weights)
    
    # 첫날 예측에 대한 중간 패널티
    first_day_penalty = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])) * 35.0
    
    # 과대 예측에 대한 패널티 (상승 예측에 더 큰 패널티)
    overprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_pred - y_true) * tf.constant([250.0, 220.0, 200.0, 180.0, 160.0], dtype=tf.float32)
    )
    
    # 과소 예측에 대한 패널티 (하락 예측에 더 큰 패널티)
    underprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_true - y_pred) * tf.constant([250.0, 220.0, 200.0, 180.0, 160.0], dtype=tf.float32)
    )
    
    # 추세 손실 (하락 추세에 더 민감하게)
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    trend_weights = tf.constant([0.3, 0.25, 0.25, 0.2], dtype=tf.float32)
    trend_loss = tf.reduce_mean(tf.square(y_true_diff - y_pred_diff) * trend_weights * 30.0 + epsilon)
    
    # 방향성 손실 (하락 방향에 더 민감하게)
    direction_weights = tf.constant([0.3, 0.25, 0.25, 0.2], dtype=tf.float32)
    direction_loss = tf.reduce_mean(
        tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)) * direction_weights * 40.0 + epsilon
    )
    
    # 연속성 손실 (부드러운 변화에 중점)
    continuity_weights = tf.constant([0.3, 0.25, 0.25, 0.2], dtype=tf.float32)
    continuity_loss = tf.reduce_mean(
        tf.square(y_pred[:, 1:] - y_pred[:, :-1] - (y_true[:, 1:] - y_true[:, :-1])) * continuity_weights * 25.0
    )
    
    # 장기 예측 정확도 향상을 위한 추가 손실
    long_term_loss = tf.reduce_mean(tf.square(y_true[:, -1] - y_pred[:, -1])) * 45.0
    
    # 가중치 적용
    weighted_loss = (
        base_loss +
        0.8 * first_day_penalty +
        1.1 * overprediction_penalty +
        1.1 * underprediction_penalty +
        0.8 * trend_loss +
        0.9 * direction_loss +
        0.7 * continuity_loss +
        1.2 * long_term_loss  # 장기 예측 정확도 향상
    )
    return weighted_loss

# 커스텀 레이어들
class LastPriceExtractor(Layer):
    def __init__(self, **kwargs):
        super(LastPriceExtractor, self).__init__(**kwargs)
        
    def call(self, inputs):
        return inputs[:, -1, 0:1]
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

class MarketContextLayer(Layer):
    def __init__(self, feature_dim, **kwargs):
        super(MarketContextLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.attention = MultiHeadAttention(
            num_heads=4,
            key_dim=feature_dim
        )
        
    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        return attention_output
    
    def compute_output_shape(self, input_shape):
        return input_shape

class AttentionGRU(Layer):
    def __init__(self, units, **kwargs):
        super(AttentionGRU, self).__init__(**kwargs)
        self.units = units
        self.gru = GRU(units, return_sequences=True)
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units)
        self.add = tf.keras.layers.Add()
        
    def build(self, input_shape):
        self.gru.build(input_shape)
        self.attention.build(
            query_shape=(input_shape[0], input_shape[1], self.units),
            key_shape=(input_shape[0], input_shape[1], self.units),
            value_shape=(input_shape[0], input_shape[1], self.units)
        )
        super(AttentionGRU, self).build(input_shape)
        
    def call(self, inputs, training=None):
        gru_output = self.gru(inputs, training=training)
        attention_output = self.attention(
            query=gru_output,
            key=gru_output,
            value=gru_output,
            training=training
        )
        output = self.add([gru_output, attention_output])
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

# 기본 모델 빌더
def build_base_model(input_shape, output_days=5):
    # 입력 레이어
    inputs = Input(shape=input_shape)
    
    # Multi-scale Convolutional Input (MCI)
    conv_outputs = []
    kernel_sizes = [2, 3, 5, 7, 11]  # 더 다양한 시간 스케일
    for kernel_size in kernel_sizes:
        conv = Conv1D(128, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
        conv = BatchNormalization()(conv)
        conv_outputs.append(conv)
    
    # 컨볼루션 출력 결합
    x = Concatenate()(conv_outputs)
    
    # Attention 메커니즘 강화
    attention_output = MultiHeadAttention(
        num_heads=8,
        key_dim=128
    )(x, x)
    
    # Residual connection
    x = tf.keras.layers.Add()([x, attention_output])
    
    # GRU 레이어 강화
    gru_output = GRU(256, return_sequences=True)(x)
    gru_output = Dropout(0.3)(gru_output)
    
    # 추가 Attention 레이어
    attention_output2 = MultiHeadAttention(
        num_heads=8,
        key_dim=256
    )(gru_output, gru_output)
    
    # Residual connection
    x = tf.keras.layers.Add()([gru_output, attention_output2])
    
    # 시퀀스의 마지막 타임스텝만 선택
    x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x)
    
    # Dense 레이어
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 출력 레이어 (5% 제한을 위한 tanh 활성화 함수 사용)
    outputs = Dense(output_days, activation='tanh')(x) * 0.05  # tanh의 출력 범위를 -0.05에서 0.05로 조정
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=outputs)
    
    # 컴파일
    optimizer = Adam(learning_rate=0.0003)  # 더 안정적인 학습을 위해 학습률 감소
    model.compile(
        optimizer=optimizer,
        loss=enhanced_weighted_time_mse,
        metrics=['mae'],
        jit_compile=True
    )
    
    return model

# 앙상블 모델 클래스
class EnsembleModel:
    def __init__(self, input_shape, num_models=3):
        self.input_shape = input_shape
        self.num_models = num_models
        self.models = []
        
    def build_models(self):
        for i in range(self.num_models):
            model = build_base_model(self.input_shape)
            self.models.append(model)
    
    def train(self, X_train, y_train, X_val, y_val, scaler):
        histories = []
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.num_models}")
            history = self._train_single_model(model, X_train, y_train, X_val, y_val)
            histories.append(history)
            
            # 모델과 스케일러 저장
            save_model_and_scaler(model, scaler, i+1)
            print(f"Model {i+1} and scaler saved successfully")
        
        return histories
    
    def _train_single_model(self, model, X_train, y_train, X_val, y_val):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=120,  # 인내심 조정
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.15,
                patience=25,  # 인내심 조정
                min_lr=1e-7,
                min_delta=0.0001
            ),
            ModelCheckpoint(
                f'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=450,  # 에포크 수 조정
            batch_size=40,  # 배치 크기 조정
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

def get_latest_stock_price(stock_name):
    """데이터베이스에서 가장 최근 주가를 가져오는 함수"""
    query = """
    SELECT close_price, time
    FROM stock_prices
    WHERE stock_name = %s
    ORDER BY time DESC
    LIMIT 1
    """
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("데이터베이스 연결 실패")
        
        with conn.cursor() as cur:
            cur.execute(query, (stock_name,))
            result = cur.fetchone()
            
            if result:
                price, time = result
                print(f"조회된 주가: {price:,.0f}원 (기준일: {time})")
                return float(price)
            else:
                # 데이터가 없는 경우 기본값 반환
                print(f"경고: {stock_name} 종목의 주가 데이터를 찾을 수 없습니다. 기본값을 사용합니다.")
                return 81800.0  # 기본값으로 최근 종가 사용
                
    except Exception as e:
        print(f"최근 주가 조회 중 오류 발생: {e}")
        # 오류 발생 시 기본값 반환
        return 81800.0  # 기본값으로 최근 종가 사용
    finally:
        if conn:
            conn.close() 