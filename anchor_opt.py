import tensorflow as tf
from data import dataset_coco
from data.kmeans import kmeans, avg_iou
from tqdm import tqdm

def load_dataset(tfrecord_dir):
    train_dataset = dataset_coco.prepare_evalloader(
        tfrecord_dir=tfrecord_dir,
        img_size=256,
        subset='train'
    )

    anchor_size_dataset = []
    for img, labels in tqdm(train_dataset):
        num_obj = labels['num_obj'][0]
        bboxes = labels['bbox'][0, :num_obj]

        for bbox in bboxes:
            [ymin, xmin, ymax, xmax] = bbox.numpy() / 256
            anchor_size_dataset.append([xmax - xmin, ymax - ymin])

    return np.array(anchor_size_dataset)

if __name__ == '__main__':
    dataset = load_dataset('data/obj_tfrecord_256x256_20201102')
    out = kmeans(data, k=3)
    
    print(f"Avg IOU: {avg_iou(data, out) * 100:.2f}%")
    print(f"Boxes:\n {out}")
    print(f'{data.shape[0]} Objects is calculated.')
    