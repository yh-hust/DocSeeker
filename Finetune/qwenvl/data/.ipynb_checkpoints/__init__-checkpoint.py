import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

ZKT = {
    "annotation_path": "/home/ma-user/work/wh/datasets/train_datas/zktv4/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/train_datas/zktv4/images",
}

ZKT_DESC = {
    "annotation_path": "/home/ma-user/work/wh/datasets/train_datas/zkt_desc/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/train_datas/zkt_desc/images",
}

SHUXING = {
    "annotation_path": "/home/ma-user/work/wh/datasets/train_datas/shuxing3/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/train_datas/shuxing3/images",
}

CHART = {
    "annotation_path": "/home/ma-user/work/wh/datasets/source_datas/chart/qwenvl_format/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/source_datas/chart/meta_data_2_wanghao",
}

CHART_WITH_CAPTION = {
    "annotation_path": "/home/ma-user/work/wh/datasets/source_datas/chart/qwenvl_format_with_caption/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/source_datas/chart/meta_data_2_wanghao",
}

ICON = {
    "annotation_path": "/home/ma-user/work/wh/datasets/train_datas/icon_des/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/train_datas/icon_des/images",
}
Stylistic = {
    "annotation_path": "/home/ma-user/work/wh/datasets/train_datas/Stylistic3/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/train_datas/Stylistic3/images",
}
Number = {
    "annotation_path": "/home/ma-user/work/wh/datasets/train_datas/Number/train/conv.jsonl",
    "data_path": "/home/ma-user/work/wh/datasets/train_datas/Number/images",
}

data_dict = {
    "number": Number,
    "stylistic": Stylistic,
    "icon": ICON,
    "chart": CHART,
    "chart_with_caption": CHART_WITH_CAPTION,
    "shuxing": SHUXING,
    "zkt_desc": ZKT_DESC,
    "zkt": ZKT,
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
