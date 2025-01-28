import cv2
import numpy as np
from skimage.morphology import skeletonize

def divide_contour_into_segments(mask, segment_length=34, start_index=47):
    """
    Divide o maior contorno de uma máscara em segmentos.

    Args:
        mask (numpy.ndarray): Máscara binária.
        segment_length (int): Comprimento de cada segmento.
        start_index (int): Índice inicial para começar os segmentos.

    Returns:
        tuple: Skeleton, maior contorno e lista de segmentos.
    """
    mask = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(mask).astype(np.uint8)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Ajustar o ponto inicial
    start_index = start_index % len(largest_contour)  # Garante que está dentro do intervalo
    largest_contour = np.roll(largest_contour, -start_index, axis=0)  # Rotaciona o contorno

    # Dividir o contorno em segmentos
    segments = []
    for i in range(0, len(largest_contour), segment_length):
        segment = largest_contour[i:i + segment_length]
        segments.append(segment)

    return skeleton, largest_contour, segments


def find_closest_segment(x, y, segments, threshold=5.0):
    closest_segment = None
    min_distance = float('inf')
    point = np.array([x, y])

    for i, segment in enumerate(segments):
        # Extraindo apenas os pontos do segmento
        # (supondo que cada elemento de 'segment' seja algo como [ [x, y], ... ]
        #  ou [(x, y), ... ])
        segment_points = np.array([p[0] for p in segment])
        
        # Calcula a distância entre (x,y) e todos os pontos do segmento
        distances = np.linalg.norm(segment_points - point, axis=1)
        
        # Pega a menor distância desse segmento
        closest_distance = np.min(distances)
        
        # Verifica se ela é a menor de todas
        if closest_distance < min_distance:
            min_distance = closest_distance
            closest_segment = i

    # Se a menor distância total for maior que nosso threshold,
    # retornamos None, indicando que não há segmento "próximo".
    # if min_distance > threshold:
    #     return None

    # Se quiser manter seu ajuste de índice (começando em 0 a partir de 32 ou similar):
    if closest_segment is not None:
        closest_segment = (closest_segment - 47) % len(segments)

    return closest_segment

def get_closest_segment_for_point(x, y, mask_path, segment_length=34):
    """
    Retorna o índice do segmento mais próximo para um ponto (x, y) a partir de uma máscara.
    
    Args:
        x (int): Coordenada x.
        y (int): Coordenada y.
        mask_path (str): Caminho para a máscara binária.
        segment_length (int): Comprimento dos segmentos.

    Returns:
        int: Índice do segmento mais próximo.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Máscara não encontrada em {mask_path}")
    
    _, _, segments = divide_contour_into_segments(mask, segment_length)
    return find_closest_segment(x, y, segments)
