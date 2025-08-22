#!/usr/bin/env python3
import json
import os
import time
import requests

def create_test_sample(size):
    """테스트용 샘플 생성"""
    chunk_file = 'outputs/chunks_sample_1k/chunks_chunks_sample_1k.json'
    
    if not os.path.exists(chunk_file):
        print(f"청크 파일을 찾을 수 없습니다: {chunk_file}")
        return False
    
    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample_chunks = data['chunks'][:size]
    sample_data = {
        'chunks': sample_chunks,
        'count': len(sample_chunks)
    }
    
    output_dir = f'outputs/test_{size}'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/chunks_test_{size}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {size}개 샘플 생성: {output_file}")
    return True

def test_embedding_performance(size, batch_size=50):
    """임베딩 성능 테스트"""
    print(f"\n🚀 {size}개 청크 임베딩 테스트 시작 (배치 크기: {batch_size})")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            'http://localhost:8000/embedding/',
            params={
                'chunks_file': f'test_{size}',
                'content_field': 'chunk_content',
                'batch_size': batch_size,
                'embeddings_format': 'json',
                'output_name': f'test_{size}_embeddings'
            },
            timeout=300  # 5분 타임아웃
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 성공: {result['count']}개 임베딩 완료")
            print(f"⏱️  소요시간: {elapsed:.2f}초")
            print(f"🚄 처리 속도: {size/elapsed:.2f} 청크/초")
            return True
        else:
            print(f"❌ 실패: {response.status_code} - {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ 타임아웃: {size}개 처리에 5분 초과")
        return False
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return False

if __name__ == "__main__":
    test_sizes = [10, 50, 100]
    
    print("🧪 임베딩 성능 테스트 시작")
    print("=" * 50)
    
    for size in test_sizes:
        # 샘플 생성
        if create_test_sample(size):
            # 성능 테스트
            success = test_embedding_performance(size)
            
            if not success:
                print(f"❌ {size}개 테스트 실패 - 더 큰 크기 테스트 중단")
                break
                
        print("-" * 30)
    
    print("\n✨ 테스트 완료!")
