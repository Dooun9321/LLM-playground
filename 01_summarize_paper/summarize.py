from dotenv import load_dotenv
from google import genai
from google.genai import types

import os 

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def review_paper_pdf(paper_pdf_path: str, example_review_pdf_path: str):
    """
    논문 PDF를 분석하고 리뷰합니다.
    
    Args:
        paper_pdf_path: 리뷰할 논문 PDF 파일 경로
        example_review_pdf_path: 예시 리뷰 PDF 파일 경로
    
    Returns:
        논문 리뷰 텍스트
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    # PDF 파일 업로드 (파일 경로를 직접 전달)
    print(f"예시 리뷰 PDF 업로드 중: {example_review_pdf_path}")
    example_file = client.files.upload(file=example_review_pdf_path)
    
    print(f"분석할 논문 PDF 업로드 중: {paper_pdf_path}")
    paper_file = client.files.upload(file=paper_pdf_path)

    system_prompt = """
당신은 통계학 박사 출신 AI Engineer 입니다.
주어진 논문을 깊이 있게 분석하고 리뷰해주세요.

첫 번째 문서는 논문 리뷰 예시입니다. 이 예시의 구조와 스타일을 참고하여 리뷰를 작성해주세요.
두 번째 문서는 리뷰할 논문입니다.

리뷰는 다음 내용을 포함해야 합니다:
1. 논문의 핵심 아이디어와 기여도
2. 제안된 방법론 또는 알고리즘의 상세 설명
3. 주요 실험 결과 및 성능 분석
4. 논문의 강점과 한계점
5. 실무 적용 가능성 및 향후 연구 방향

예시 리뷰의 형식과 깊이를 참고하되, 리뷰할 논문의 특성에 맞게 작성해주세요.
모든 설명은 한글로 작성해주세요.
"""

    # 파일들과 함께 요청
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            system_prompt,
            example_file,
            paper_file
        ],
    )

    print("\n논문 리뷰 생성 완료!")
    return response.text


if __name__ == "__main__":
    # 예시: 논문 리뷰 실행
    example_review_path = "Temporal_Fusion_Transformer.pdf"
    paper_to_review_path = "1906.02120v2.pdf"  # 리뷰할 논문 PDF 경로를 지정하세요
    
    if os.path.exists(example_review_path) and os.path.exists(paper_to_review_path):
        review = review_paper_pdf(paper_to_review_path, example_review_path)
        print("\n" + "="*80)
        print("논문 리뷰:")
        print("="*80)
        print(review)
        
        # 리뷰를 파일로 저장
        output_file = paper_to_review_path.replace(".pdf", "_review.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(review)
        print(f"\n리뷰가 {output_file}에 저장되었습니다.")
    else:
        print("PDF 파일 경로를 확인해주세요:")
        print(f"예시 리뷰: {example_review_path} (존재: {os.path.exists(example_review_path)})")
        print(f"리뷰할 논문: {paper_to_review_path} (존재: {os.path.exists(paper_to_review_path)})")