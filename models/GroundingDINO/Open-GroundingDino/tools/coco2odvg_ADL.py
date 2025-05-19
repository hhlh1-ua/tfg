import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO

# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}

id_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46}
key_list=list(id_map.keys())
val_list=list(id_map.values())

def dump_label_map(output="./out.json"):
    ori_map = {
    "0": "bed",
    "1": "monitor",
    "2": "keyboard",
    "3": "shoe",
    "4": "blanket",
    "5": "towel",
    "6": "door",
    "7": "electric_keys",
    "8": "soap_liquid",
    "9": "container",
    "10": "tap",
    "11": "person",
    "12": "tooth_paste",
    "13": "dent_floss",
    "14": "bottle",
    "15": "trash_can",
    "16": "tooth_brush",
    "17": "laptop",
    "18": "dish",
    "19": "knife/spoon/fork",
    "20": "vacuum",
    "21": "kettle",
    "22": "pan",
    "23": "mug/cup",
    "24": "fridge",
    "25": "oven/stove",
    "26": "food/snack",
    "27": "tea_bag",
    "28": "thermostat",
    "29": "washer/dryer",
    "30": "detergent",
    "31": "cloth",
    "32": "basket",
    "33": "cell",
    "34": "book",
    "35": "comb",
    "36": "tv",
    "37": "pitcher",
    "38": "microwave",
    "39": "pills",
    "40": "shoes",
    "41": "tv_remote",
    "42": "cell_phone",
    "43": "large_container",
    "44": "milk/juice",
    "45": "perfume",
    "46": "mop"
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
