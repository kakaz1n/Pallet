import numpy as np
from threading import Thread, Lock
from queue import Queue, Empty
import sys
import os
from tqdm import tqdm
import cv2 as cv
import torch
import glob
import json
import paho.mqtt.publish as publish
import time
import numpy as np
from threading import Thread, Lock
from queue import Queue
import cv2 as cv
from skimage.morphology import skeletonize
from shapely.geometry import Polygon
from segmento import get_closest_segment_for_point
from collections import defaultdict

np.set_printoptions(threshold=sys.maxsize)

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, Instances

from shapely.geometry import Polygon, Point
import random

# Dicionário para armazenar o histórico de centros
historico_centros = defaultdict(list)

# Dicionário para armazenar cores predefinidas para cada ID
cores_ids = {}
# Função para gerar uma cor aleatória
def gerar_cor_aleatoria(): #sim
    return tuple(random.randint(0, 255) for _ in range(3))
mask_path = 'mascara_sem_buffer.png'

global tracked_objects 
tracked_objects = {}
global next_id 
next_id = 1
global iou_threshold
iou_threshold = 50  # Limiar de distância para associação de IDs

class VideoStream:
    """
    
    """

    count = 0

    def __init__(self, src):
        self.cap = cv.VideoCapture(src)
        if("mp4" in src):
            self.queue = Queue(maxsize=0)
        else:
            self.queue = Queue(maxsize=1)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        VideoStream.count+=1

    def update(self):
        while not self.stopped:
            hasframe, frame = self.cap.read()
            if not hasframe:
                self.stop()
                return
            if not self.queue.full():
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.thread.join()

    def more(self):
        return not self.queue.empty()


def set_threshold(cfg, thresh): #sim
    """
    
    """
    cfg.OUTPUT_DIR=".\\output_pecas_dataset_368images_4classes_cam2lab10"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh


def run_inference(cfg): #sim
    """
    
    """
    predictor = DefaultPredictor(cfg)
    return predictor


def set_cfg(train, workers, n_classes, ipb, lr, iterations, bspi, device, config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"), weights = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")):
    """
    
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = train
    cfg.DATASETS.TEST = []
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.MODEL.WEIGHTS = weights
    cfg.SOLVER.IMS_PER_BATCH = ipb
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.STEPS = []      
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bspi
    if(torch.cuda.device_count() >= 1):
        device = "cuda"
    cfg.MODEL.DEVICE = device
    return cfg

def load_train(filename, thresh): ##sim
    """
    
    """
    cfg = get_cfg()
    cfg.merge_from_file(filename)
    set_threshold(cfg, thresh)  #aqui está a pasta dos pesos
    model = build_model(cfg)  # Reconstrói o modelo com a estrutura corrigida
    # model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device('cpu')))   #Usar após treinamento da NN
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # Carrega apenas o state_dict do modelo, isso é utilizado no modo load
    predictor = run_inference(cfg)
    return predictor, cfg


def get_dicts(img_dir, class_names):
    """
    
    """

    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []

        for _, anno in annos.items():
            shape_attr = anno["shape_attributes"]
            region_attr = anno["region_attributes"]
            px = shape_attr["all_points_x"]
            py = shape_attr["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            class_name = region_attr["name"]
            category_id = class_names.index(class_name)

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def box_to_polygon(box):
    """
    
    Parameters:

    Returns:

    Raises:
    """

    x1, y1, x2, y2 = box
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def hierarchically_detection(bandejas_ord, image, window): #sim

    """
    Detecta hierarquicamente os objetos i.e. Considerando os objetos como se
    fossem polígonos baseados nas caixas delimitadoras o algoritmo consiste
    em agrupar um ou mais polígonos dentro de outro(s).

    A idéia é análoga a uma cesta de produtos onde uma cesta pode conter um
    ou mais produto(s) e dentro da embalagem do(s) produtos podem haver um ou
    mais itens consecutivamente.

    Inicialmente é verificado se a lista é ou não vazia, caso seja então dois objetos são iguais ou não, caso sejam

    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=x.ndim)``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : ndarray, shape Q

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    numpy.tensordot, tensorinv, numpy.einsum
    """

    band = []
    nao_contem=0
    tipos = []
    dict = {}
    adicionou_bandeja=False

    i = 0

    if(len(bandejas_ord) == 0):
        pass
    else:
        while(len(bandejas_ord) != 1):
            j = 0
            while(j != len(bandejas_ord)): #for j in range(0, len(bandejas_ord)):

                if(i == j):
                    j+=1
                    pass
                elif(len(bandejas_ord) < 2):
                    break
                else:   
                    id = bandejas_ord[i][0]
    
                    pol1 = bandejas_ord[i][1][0]
                    obj1 = bandejas_ord[i][1][1]
                    center1 = bandejas_ord[i][1][2]

                    pol2 = bandejas_ord[j][1][0]
                    obj2 = bandejas_ord[j][1][1]
                    # center2 = bandejas_ord[j][1][2]

                    #print(f"bandejas_ord:{bandejas_ord}\nbandeja1:{bandejas_ord[i]}\n bandeja[2]:{bandejas_ord[j]}")

                    if((obj1 == "pallet" and obj2 == "pallet") or (obj1 == "template" and obj2 == "template") or (obj1 == "obj_u1" and obj2 == "obj_u1")):
                        j+=1
                        pass
                    else:
                        if(pol1.intersects(pol2)):
                            tipos.append(obj1)
                            tipos.append(obj2)
                            dict["tipo"] = list(set(tipos))
                            dict["id"] = id
                            x, y = int(center1[0]), int(center1[1]) # x, y = int(centers[mask][i][0]), int(centers[mask][i][1])
                            posicao = {"x": x, "y": y}
                            dict["posicao"] = posicao
                            dict["camera"] = window
                            #print(f"\n\nintersects band:{band}")
                            band.append(dict)
                            # if(Point(center1[0], center1[1]).within(conveyor_pts) and Point(center2[0], center2[1]).within(conveyor_pts)):
                            #     band_dentro.append(dict)
                            # else:
                            #     band_fora.append(dict)
                            #print(f"\nintersects band:{band}")
                            #print(f"\nbandejas_ord:{bandejas_ord}")
                            bandejas_ord.remove(bandejas_ord[j])
                            #print(f"\nbandejas_ord:{bandejas_ord}")
                            adicionou_bandeja=True
                            text = f"ID: {id}, Center: ({x}, {y})"
                            #text = f"Center: ({x}, {y})" 
                            tipos = []
                            dict = {}
                            #descomentar depois abaixo
                            image = cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                            cv.circle(image,(x,y),5,(0, 255, 0))
                        else:
                            j+=1
                            nao_contem+=1

            if(adicionou_bandeja == True and len(bandejas_ord) != 1):
                #print(f"\n\nbandejas_ord segundo if:{bandejas_ord}\n")
                bandejas_ord.remove(bandejas_ord[i])
                #print(f"bandejas_ord segundo if:{bandejas_ord}\n")
                j = 0
                adicionou_bandeja=False
            elif(len(bandejas_ord) == 1):
                #print(f"\n\nbandejas_ord elif:{bandejas_ord}\n")
                pass
            else:
                tipos.append(obj1)
                dict["tipo"] = list(set(tipos))
                dict["id"] = id
                x, y = int(center1[0]), int(center1[1])
                posicao = {"x": x, "y": y}
                dict["posicao"] = posicao
                dict["camera"] = window
                #print(f"\n\nelse band:{band}\n")
                band.append(dict)
                # if(Point(center1[0], center1[1]).within(conveyor_pts) and Point(center2[0], center2[1]).within(conveyor_pts)):
                #     band_dentro.append(dict)
                # else:
                #     band_fora.append(dict)
                #print(f"else band:{band}\n")
                #print(f"else bandejas_ord:{bandejas_ord}\n")
                bandejas_ord.remove(bandejas_ord[i])
                #print(f"else bandejas_ord:{bandejas_ord}\n")
                text = f"ID: {id}, Center: ({x}, {y})" 
                #text = f"Center: ({x}, {y})" 
                tipos = []
                dict = {}
                #descomentar depois abaixo
                image = cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                cv.circle(image,(x,y),5,(255, 0, 0))
    if(len(bandejas_ord) == 0):
        pass
    else:
        id = bandejas_ord[i][0]

        pol1 = bandejas_ord[i][1][0]
        obj1 = bandejas_ord[i][1][1]
        center1 = bandejas_ord[i][1][2]

        tipos.append(obj1)
        dict["tipo"] = list(set(tipos))
        dict["id"] = id
        x, y = int(center1[0]), int(center1[1])
        posicao = {"x": x, "y": y}
        dict["posicao"] = posicao
        dict["camera"] = window
        #print(f"\n\noutside band:{band}\n")
        band.append(dict)
        # if(Point(center1[0], center1[1]).within(conveyor_pts) and Point(center2[0], center2[1]).within(conveyor_pts)):
        #     band_dentro.append(dict)
        # else:
        #     band_fora.append(dict)
        #print(f"outside band:{band}\n")
        text = f"ID: {id}, Center: ({x}, {y})" 
        #text = f"Center: ({x}, {y})" 
        tipos = []
        dict = {}
        #descomentar depois abaixo
        image = cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.circle(image,(x,y),5,(0, 0, 255))

    return band, image


def group_objects(band): #sim
    """
    
    """

    grouped = {}

    for item in band:
        id = item["id"]
        pos = item["posicao"]
        cam = item["camera"]
        coord = (pos["x"], pos["y"])
        
        # Se as coordenadas ainda não estiverem no dicionário agrupados, cria uma nova entrada
        if coord not in grouped:
            grouped[coord] = {
                'tipo': set(),
                'id': set(),
                'posicao': pos,
                'camera': cam
            }
            # grouped[coord] = {
            #     "tipo": set(),
            #     "posicao": pos,
            #     "camera": cam
            # }
        
        # Adiciona os dados ao grupo existente
        grouped[coord]["tipo"].update(item["tipo"])
        grouped[coord]["id"].add(item["id"])

    # Prepara a lista de resultados no formato desejado
    # msg = {'bandeja': []}
    msg = []
    for data in grouped.values():
        # msg['bandeja'].append({
        #     'id': list(data['id'])[0],  # A chave 'id' deve ter um único valor
        #     'posicao': data['posicao'],
        #     'tipo': list(data['tipo']),
        #     'camera': data['camera']
        # })
        msg.append({
            "id": list(data["id"])[0],  # A chave 'id' deve ter um único valor
            "posicao": data["posicao"],
            "tipo": list(data["tipo"]),
            "camera": data["camera"]
        })
        # msg.append({
        #     "posicao": data["posicao"],
        #     "tipo": list(data["tipo"]),
        #     "camera": data["camera"]
        # })
    
    return msg

def resizing(image, video_stream, w, h):
    """
    
    """

    # Convert Matplotlib RGB format to OpenCV BGR format
    visualization = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            
    width = int(video_stream.cap.get(cv.CAP_PROP_FRAME_WIDTH) * w)
    height = int(video_stream.cap.get(cv.CAP_PROP_FRAME_HEIGHT) * h)

    visualization_resized = cv.resize(visualization, (width, height))

    return visualization, visualization_resized

def create_and_move_window(cam, img, x, y):
    """
    
    """

    #código disponível em: https://stackoverflow.com/questions/47234590/cv2-imshow-image-window-placement-is-outside-of-viewable-screen/47235549#47235549
    cv.namedWindow(str(cam))
    # cv.moveWindow(str(cam), x, y)
    # cv.imshow(str(cam),img)


next_id = 1

visited_windows = list()
id_lock = Lock()
messages = list()

#def process_stream(cfg, stream, predictor, video_writer, metadata, num_frames, window_name):
def process_stream(cfg, stream, predictor, metadata, num_frames, num_cameras, window_name): #sim
    """
    
    """


    global next_id
    global visited_windows
    global messages

    """Função para processar cada stream de vídeo em uma thread separada"""
    #for visualization, visualization_resized in tqdm(display_live_one_video(cfg, stream, predictor, metadata, num_frames), total=num_frames):
    for message, visualization_resized in tqdm(display_live_one_video(cfg, stream, predictor, metadata, num_frames, window_name), total=num_frames):
        #2 = camera do rack
        #1 = camera da fresa
        
        #video_writer.write(visualization)  # Exibir o frame do vídeo
        if(window_name == 2):
            create_and_move_window(window_name, visualization_resized, 0, 10)
            create_and_move_window("Teste 1", visualization_resized, 800, 0)
        else:
            # create_and_move_window(window_name, visualization_resized, 730, 10)
            create_and_move_window(window_name, visualization_resized, 0, 450)
        # message_queue.put(message)
        # print("message_queue.put(message)")

        with id_lock:
            if(len(visited_windows) == 1):
                message = [{**item, 'id': i} for i, item in enumerate(message, start=1)]
            else:
                message = [{**item, 'id': i} for i, item in enumerate(message, start=next_id+1)]

        messages.append(message)

        next_id = len(message)


        if(len(visited_windows) == num_cameras):
            next_id = 0
            messages = []
            visited_windows = []


def distancia_ciclica(a, b, max_val=80): #sim
    diff = abs(a - b)
    # Se vai de 0..80, então há (max_val + 1) posições possíveis
    return min(diff, (max_val + 1) - diff)

# Código adaptado de: https://stackoverflow.com/questions/60663073/how-can-i-properly-run-detectron2-on-videos
#def display_video(cap, predictor, prev_frame_time, new_frame_time):
def display_live_one_video(cfg, video_stream, predictor, metadata, maxFrames, cam, frame_skip=1, segments=None): #sim
    """
    Processa o vídeo ao vivo com detecção e segmentação.
    Adiciona o número do segmento mais próximo do objeto identificado.
    """
    global tracked_objects
    readFrames = 1
    global next_id
    global cores_ids
    historico_segmentos = defaultdict(list)
    stale_counts = {}
    while 1:

        while not video_stream.more():
            print("Loading... please wait!")
            if video_stream.more():
                display_live_one_video(cfg, video_stream, predictor, metadata, maxFrames, cam, frame_skip, segments)
                break
            
        while video_stream.more():
            visited_windows.append(cam)
            visited_windows.sort()

            # print(f"visited windows before: {visited_windows}")

            frame = video_stream.read()
            if frame is None:
                break
            # Dentro do loop de processamento de frames:
            
            height, width, _ = frame.shape
            imagem_rastros_persistente = np.zeros((height, width, 3), dtype=np.uint8)
            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes
            classes = instances.pred_classes
            centers = boxes.get_centers().numpy()

            class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            class_labels = [class_names[i] for i in classes]

            visualizer = Visualizer(frame[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
            image = visualizer.draw_instance_predictions(instances).get_image()

            bandejas = [
                (box_to_polygon(box), label, center)
                for box, label, center in zip(boxes.tensor.cpu().numpy(), class_labels, centers)
            ]

            ordem = {"pallet": 1, "template": 2, "obj_u1": 3, "obj_u2": 4}
            bandejas_ord = list(enumerate(sorted(bandejas, key=lambda x: ordem[x[1]]), start=1))

            band, image = hierarchically_detection(bandejas_ord, image, cam)

            # print(f"\nband: {band} cam: {cam}")

            msg = group_objects(band)

            # Atualizar ou associar IDs
            # Lista para armazenar os segmentos e centros dos novos objetos
            novos_segmentos = []

            # Determinar os segmentos para os novos objetos detectados
            for obj in msg:
                center = obj["posicao"]
                obj_x, obj_y = center["x"], center["y"]
                closest_segment = get_closest_segment_for_point(obj_x, obj_y, mask_path)

                # Armazenar o segmento e o centro do novo objeto
                novos_segmentos.append({
                    "center": {"x": obj_x, "y": obj_y},
                    "segmento": closest_segment
                })
            new_tracked_objects = {}
            usados = set()  # Para garantir que não reutilizamos IDs já atribuídos neste frame

            MAX_DISTANCE = 1  # Distância cíclica máxima para considerar o mesmo ID

            for novo_obj in novos_segmentos:
                novo_segmento = novo_obj["segmento"]
                novo_center = novo_obj["center"]

                matched_id = None
                min_segment_distance = float('inf')  # Começa com infinito

                for obj_id, segmentos in historico_segmentos.items():
                    # Se esse ID já foi atribuído neste frame, pula
                    # if obj_id in usados:
                    #     continue
                    
                    # Se não tiver histórico para esse ID, pula
                    if len(segmentos) == 0:
                        continue

                    ultimo_segmento = segmentos[-1]["segmento"]

                    # Calcula a distância cíclica
                    segment_distance = distancia_ciclica(novo_segmento, ultimo_segmento, max_val=80)

                    # Somente consideramos se for menor que o min atual e dentro do limite MAX_DISTANCE
                    if segment_distance < min_segment_distance and segment_distance <= MAX_DISTANCE:
                        min_segment_distance = segment_distance
                        matched_id = obj_id

                if matched_id is not None:
                    # Achou um ID existente para este novo objeto
                    new_tracked_objects[matched_id] = {
                        "posicao": novo_center,
                        "segmento": novo_segmento,
                        "segmentoMax": 80
                    }
                    usados.add(matched_id)
                    novo_obj["id"] = matched_id
                else:
                    # Não achou nenhum ID próximo (<= 5), então cria um novo
                    new_tracked_objects[next_id] = {
                        "posicao": novo_center,
                        "segmento": novo_segmento,
                        "segmentoMax": 80
                    }
                    novo_obj["id"] = next_id
                    usados.add(next_id)
                    matched_id = next_id
                    next_id += 1
                if matched_id not in historico_segmentos:
                    historico_segmentos[matched_id] = []
                    historico_centros[matched_id] = []
                    stale_counts[matched_id] = 0 
                # Atualiza histórico do matched_id
                historico_segmentos[matched_id].append(novo_obj)
                historico_centros[matched_id].append((float(novo_center["x"]), float(novo_center["y"])))

            # Substituir objetos rastreados pelos atualizados
            for obj_id in tracked_objects.keys():
                if obj_id not in usados:
                    stale_counts[obj_id] += 1
                else:
                    # se o ID foi usado, zera a contagem
                    stale_counts[obj_id] = 0
            # Remover IDs cuja contagem ultrapassou 10
            ids_para_remover = [obj_id for obj_id, count in stale_counts.items() if count >= 3]

            for obj_id in ids_para_remover:
                del stale_counts[obj_id]
                if obj_id in new_tracked_objects:
                    del new_tracked_objects[obj_id]
                if obj_id in historico_segmentos:
                    del historico_segmentos[obj_id]
                if obj_id in historico_centros:
                    del historico_centros[obj_id]
            print(stale_counts)
            # Finalmente, atualiza tracked_objects
            tracked_objects = new_tracked_objects
            # Desenhar os últimos 5 centros de cada ID na imagem
            # Gerar cores para novos IDs
            for obj_id in historico_centros.keys():
                if obj_id not in cores_ids:
                    cores_ids[obj_id] = gerar_cor_aleatoria()

            # Desenhar os últimos 5 centros de cada ID na imagem
            for obj_id, centros in historico_centros.items():
                cor = cores_ids[obj_id]  # Agora nunca dará KeyError
                for cx, cy in centros[-5:]:
                    cv.circle(image, (int(cx), int(cy)), 3, cor, -1)  # Desenhar círculo

            for obj_id, centros in historico_centros.items():
                cor = cores_ids[obj_id]

                # Pegue apenas os últimos 5 pontos
                ultimos_centros = centros[-5:]  # lista com até 5 elementos

                # Desenha as linhas entre esses últimos pontos
                for i in range(1, len(ultimos_centros)):
                    pt1 = (int(ultimos_centros[i - 1][0]), int(ultimos_centros[i - 1][1]))
                    pt2 = (int(ultimos_centros[i][0]), int(ultimos_centros[i][1]))
                    cv.line(imagem_rastros_persistente, pt1, pt2, cor, thickness=2)

                # Desenha pequenos círculos em cada um desses últimos 5 pontos
                for (cx, cy) in ultimos_centros:
                    cv.circle(imagem_rastros_persistente, (int(cx), int(cy)), 3, cor, -1)

                # Opcional: desenha o ID ao lado do último ponto
                if len(ultimos_centros) > 0:
                    cx, cy = ultimos_centros[-1]
                    cv.putText(imagem_rastros_persistente, f"ID: {obj_id}",
                            (int(cx) + 5, int(cy) - 5),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv.LINE_AA)
            imagem_rastros_persistente = cv.resize(imagem_rastros_persistente, (800, 800))
            # cv.imshow("Rastros Acumulados - Últimos 5 Pontos de Cada ID", imagem_rastros_persistente)

            # Imprimir o histórico de segmentos para cada ID
            # print("Histórico de Segmentos por ID:")
            # for obj_id, segmentos in historico_segmentos.items():
            #     print(f"ID {obj_id}: {segmentos}")

            # Exibir objetos rastreados formatados
            formatted_tracked_objects = sorted(
                [
                    {
                        "idPallet": obj_id,
                        # "posicao": obj_data["posicao"],
                        "segmento": obj_data["segmento"],
                        "segmentoMax": obj_data.get("segmentoMax", None)
                    }
                    for obj_id, obj_data in tracked_objects.items()
                ],
                key=lambda obj: int(obj["idPallet"])
            )
            data = {
                "quantity" : len(formatted_tracked_objects),
                "pallets" : formatted_tracked_objects
            }
            print("Objetos rastreados formatados:")
            data = json.dumps(data, indent=4, ensure_ascii=False)
            print(data)
            # objetos_json = json.dumps(formatted_tracked_objects, indent=4, ensure_ascii=False)
            vis, visres = resizing(image, video_stream, 0.5, 0.5)
            publish.single("projetoFinep/openLab_1.09/PalletManager/", str(data),
                            hostname="10.83.146.40", port=1884,
                            auth={'username': "planta", 'password': "Senai@UserPlanta109"})
            if cv.waitKey(1) == ord('q'):
                break

            # yield msg, visres

            while not video_stream.more():
                display_live_one_video(cfg, video_stream, predictor, metadata, maxFrames, cam, frame_skip, segments)

            readFrames += 1

        

def main():

    if os.name == "nt":
        #default_path = "C:\\Users\\Instrutor\\Downloads\\"
        default_path = ".\\"
        #json_path = ".\\pecas_dataset_camera_fresa_e_rack_pallet\\pecas\\"
        json_path = ".\\pecas_dataset_368images_4classes_cam2lab10\\pecas\\"
    else:
        default_path = "./"
        json_path = "./pecas_dataset_camera_fresa_e_rack_pallet/pecas/"
        # json_path = "./pecas_dataset_293images_cam2lab10/pecas/"

    class_names = ["pallet", "template", "obj_u1", "obj_u2"]
    # class_names = ["pallet"]
    name = "pecas"    
    thresh = 0.5

    #filename = "config_model_with_388_images_faster_rcnn_R_50_FPN_3x_gpu_4classes.yaml"
    filename = "config_model_with_368_images_faster_rcnn_R_50_FPN_3x_cuda_4classes.yaml"

   
    for d in ["train", "val"]:
        # print("for d in ['train', 'val']:")
        DatasetCatalog.register(name + "_" + d, lambda d=d: get_dicts(json_path + d, class_names))
        # print("DatasetCatalog.register(name + '_' + d, lambda d=d: get_dicts(json_path + d, class_names))")
        MetadataCatalog.get(name + "_" + d).set(thing_classes=class_names)

    metadata = MetadataCatalog.get(name + "_" + "train")
    print(metadata)
    #dataset_dicts = get_dicts(img_dir, class_names)
    

    url = "rtsp://109@191.101.234.84:8554/lab10/02?user=myuser&pass=mypass" 
    
    streams = [VideoStream(url)]
    num_cameras = len(streams)

    # width = int(streams[0].cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(streams[0].cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # frames_per_second = streams[0].cap.get(cv.CAP_PROP_FPS)

    predictor, cfg = load_train(filename, thresh)
    


    stop = False
    loop = False #Executar ou não em loop o vídeo gravado
    threads = []
    all_messages = []
    message_queue = Queue()

    # next_id = 1
    # id_lock = Lock()

    for i, stream in enumerate(streams):
        num_frames = int(stream.cap.get(cv.CAP_PROP_FRAME_COUNT))

        # print(f"\nwidth: {width}")
        # print(f"height: {height}")
        # print(f"fps: {frames_per_second}")
        # print(f"num_frames: {num_frames}\n")

        window_name = i+1
        #thread = Thread(target=process_stream, args=(cfg, stream, predictor, video_writer, MetadataCatalog.get(name + "_" + "train"), num_frames, window_name))
        thread = Thread(target=process_stream, args=(cfg, stream, predictor, metadata, num_frames, num_cameras, window_name))
        threads.append(thread)
        thread.start()
    
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
    
    # Esperar que todas as threads terminem
    for thread in threads:
        thread.join()

   
if(__name__ == "__main__"):
    main()