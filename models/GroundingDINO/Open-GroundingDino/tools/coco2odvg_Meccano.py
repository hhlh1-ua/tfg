import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO

# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}

id_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19}
key_list=list(id_map.keys())
val_list=list(id_map.values())

def dump_label_map(output="./out.json"):
    ori_map ={
        "0": "booklet",
        "1": "gray_angled_perforated_bar",
        "2": "partial_model",
        "3": "white_angled_perforated_bar",
        "4": "wrench",
        "5": "screwdriver",
        "6": "gray_perforated_bar",
        "7": "wheels_axle",
        "8": "red_angled_perforated_bar",
        "9": "red_perforated_bar",
        "10": "rod",
        "11": "handlebar",
        "12": "screw",
        "13": "tire",
        "14": "rim",
        "15": "washer",
        "16": "red_perforated_junction_bar",
        "17": "red_4_perforated_junction_bar",
        "18": "bolt",
        "19": "roller"
    }
    new_map = {}
    for key, value in ori_map.items():
        label = int(key)
        ind=val_list.index(label)
        label_trans = key_list[ind]
        new_map[label_trans] = value
    with open(output,"w") as f:
        json.dump(new_map, f)

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    x1 = round(x, 2) 
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def coco2odvg(args):
    coco = COCO(args.input) 
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']:cat['name'] for cat in cats}
    metas = []

    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            label = ann['category_id']
            category = nms[label]
            ind=val_list.index(label)
            label_trans = key_list[ind]
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": label_trans,
                "category": category
                }
            )
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                    "instances": instance_list
                }
            }
        )
    print("  == dump meta ...")
    with jsonlines.open(args.output, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    parser.add_argument("--input", '-i', required=True, type=str, help="input list name")
    parser.add_argument("--output", '-o', required=True, type=str, help="output list name")
    args = parser.parse_args()

    coco2odvg(args)
