import cv2
import tensorflow as tf
import os
import pickle


def detect_image(img):

    img = cv2.imread(img)

    with tf.gfile.FastGFile('FROZEN PATH', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

    if graph_def:
        with tf.Session() as sess:
            # Restore session
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            detected = []
            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                
                if score > 0.9:
                    x_min = bbox[1] * cols
                    y_min = bbox[0] * rows
                    x_max = bbox[3] * cols
                    y_max = bbox[2] * rows
                    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (125, 255, 51), thickness=2)
                    cv2.putText(img, 'Banana', (int(x_min), int(y_min)), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 255, 51), 1)
                    detected.append([classId, score, [x_min, y_min, x_max, y_max]])
            return img, detected


def convert_rob_pos(stri):
    con_dict = {'l0': 0, 'l1': 1, 'l2': 2, 'l3': 3, 'l4': 4,
                'm0': 5, 'm1': 6, 'm2': 7, 'm3': 8,
                'h0': 9, 'h1': 10, 'h2': 11}
    return con_dict[stri]



filelist=os.listdir('/home/frederik/Desktop/dataset_23_10_18/')
bboxs = {0: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]}, 
         1: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]},
         2: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]},
         3: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]},
         4: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]},
         5: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]},
         6: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]},
         7: {0: [1, 2, 3, 4], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4]}}


for fichier in filelist: # filelist[:] makes a copy of filelist.
    if (fichier.endswith(".jpg")):
        _, detectd = detect_image('/home/frederik/Desktop/dataset_23_10_18/' + fichier)
        xmin, ymin = detectd[0][2][0], detectd[0][2][1]
        xmax, ymax = detectd[0][2][2], detectd[0][2][3]
        print('Robot pos:', convert_rob_pos(fichier[6:8]), 'Banana pose:', fichier[4:5])
        print('(' + str(xmin) + ', ' + str(ymin) + '), (' + str(xmax) + ', ' + str(ymax) + ')')
        bboxs[int(fichier[4:5])][convert_rob_pos(fichier[6:8])][0] = xmin
        bboxs[int(fichier[4:5])][convert_rob_pos(fichier[6:8])][1] = ymin
        bboxs[int(fichier[4:5])][convert_rob_pos(fichier[6:8])][2] = xmax
        bboxs[int(fichier[4:5])][convert_rob_pos(fichier[6:8])][3] = ymax
        with open('bboxs.pkl', 'wb') as handle:
            pickle.dump(bboxs, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(bboxs)

pickle.dump(bboxs, 'bboxs.pkl')
