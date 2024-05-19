import os
import numpy as np
from pathlib import Path


def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


def create_submission(results, data_dict, base_path):
    with open("submission.csv", "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        
        for dataset in data_dict:
            if dataset in results:
                res = results[dataset]
            else:
                res = {}
            
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}
                    
                for image in data_dict[dataset][scene]:
                    image_path = str(Path(image).relative_to(base_path))
                    if image in scene_res:
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                        # print(">>", f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")


def parse_sample_submission(data_path, IMC_PATH):
    data_dict = {}
    with open(data_path, "r") as f:
        for i, l in enumerate(f):
            if i == 0:
                print("header:", l)

            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(',')
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(os.path.join(IMC_PATH, image_path))

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    return data_dict


if __name__ == "__main__":
    dataset_dir = "/home/ron/Documents/ImageMatching/image-matching-challenge-2024"
    data_dict = parse_sample_submission(
        f"{dataset_dir}/sample_submission.csv",
        dataset_dir,
    )
    dummy_predict = {
        'church': {
            'church': {
                dataset_dir + '/test/church/images/00046.png': {'R': np.zeros([3, 3]) + 1, 't': np.zeros([3, 1])},
                dataset_dir + '/test/church/images/00090.png': {'R': np.zeros([3, 3]) + 2, 't': np.zeros([3, 1])},
            }
        }
    }
    print(data_dict)
    create_submission(dummy_predict, data_dict, dataset_dir)