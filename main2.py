from model import process_images 
from model import calculate_similarity


# 이미지 처리를 위한 폴더 경로
directory = "./pokemon/images/images"

# 이미지 처리를 위한 함수 호출
fe, features, img_paths = process_images(directory)

# 이미지 유사도 계산 및 시각화를 위한 함수 호출
query_image_path = "./pokemon/images/images/doduo.png"
calculate_similarity(fe, features, img_paths, query_image_path)