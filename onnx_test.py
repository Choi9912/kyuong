#!/usr/bin/env python3
import time
import numpy as np
from app.services.embedder import Embedder
from app.config.settings import settings

def test_onnx_performance():
    """ONNX 임베딩 성능 직접 테스트"""
    print("🔧 ONNX 임베딩 성능 테스트")
    print("=" * 40)
    
    # Embedder 초기화
    print("📋 설정 정보:")
    print(f"   - 모델: {settings.embedding_model}")
    print(f"   - ONNX 경로: {settings.onnx_model_path}")
    print(f"   - 출력 차원: {getattr(settings, 'embedding_output_dim', 768)}")
    
    print("\n🚀 Embedder 초기화 중...")
    start_init = time.time()
    
    try:
        embedder = Embedder()
        end_init = time.time()
        print(f"✅ 초기화 완료: {end_init - start_init:.2f}초")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 테스트 텍스트들
    test_texts = [
        "안녕하세요. 이것은 테스트 문장입니다.",
        "ONNX 모델의 성능을 테스트하고 있습니다.",
        "임베딩 벡터가 올바르게 생성되는지 확인합니다.",
        "한국어 임베딩 모델의 처리 속도를 측정합니다.",
        "이 문장은 다섯 번째 테스트 문장입니다."
    ]
    
    # 단일 임베딩 테스트
    print(f"\n🧪 단일 임베딩 테스트 (1개 텍스트)")
    start_single = time.time()
    
    try:
        single_result = embedder.embed_texts([test_texts[0]])
        end_single = time.time()
        
        print(f"✅ 성공: {end_single - start_single:.3f}초")
        print(f"   - 벡터 차원: {len(single_result[0])}")
        print(f"   - 벡터 타입: {type(single_result[0])}")
        print(f"   - 처리 속도: {1/(end_single - start_single):.2f} 텍스트/초")
    except Exception as e:
        print(f"❌ 단일 임베딩 실패: {e}")
        return
    
    # 배치 임베딩 테스트
    print(f"\n🧪 배치 임베딩 테스트 ({len(test_texts)}개 텍스트)")
    start_batch = time.time()
    
    try:
        batch_result = embedder.embed_texts(test_texts)
        end_batch = time.time()
        
        print(f"✅ 성공: {end_batch - start_batch:.3f}초")
        print(f"   - 결과 개수: {len(batch_result)}")
        print(f"   - 처리 속도: {len(test_texts)/(end_batch - start_batch):.2f} 텍스트/초")
    except Exception as e:
        print(f"❌ 배치 임베딩 실패: {e}")
        return
    
    # 더 큰 배치 테스트
    large_texts = test_texts * 20  # 100개 텍스트
    print(f"\n🧪 대용량 배치 테스트 ({len(large_texts)}개 텍스트)")
    start_large = time.time()
    
    try:
        large_result = embedder.embed_texts(large_texts)
        end_large = time.time()
        
        print(f"✅ 성공: {end_large - start_large:.3f}초")
        print(f"   - 결과 개수: {len(large_result)}")
        print(f"   - 처리 속도: {len(large_texts)/(end_large - start_large):.2f} 텍스트/초")
    except Exception as e:
        print(f"❌ 대용량 배치 실패: {e}")
    
    print("\n✨ ONNX 성능 테스트 완료!")

if __name__ == "__main__":
    test_onnx_performance()
