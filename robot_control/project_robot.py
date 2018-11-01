import urx
from random import randint
import numpy as np
import tensorflow as tf
import cv2


class ProjectRobot:
    """ Class to represent robot with project-specific functions.
    
    Attributes:
    """

    def __init__(self, address="169.254.9.171", tcp=(0,0,0.12,0,0,0), acc=1, vel=1):
        """ ProjectRobot
        Arguments:
        """
        self.robot = urx.Robot(address)
        self.robot.set_tcp(tcp)
        self.robot.set_payload(0.5, (0,0,0))
        self.accelaration = acc
        self.velocity = vel
        self.graph_def = None
        self.waypoint = [-1.6616261926776854, -2.516756425648164, -1.7162176543532885, -2.035364287737379, -3.275768284901913, 0.021001491282445117]
        self.waypoint2 = [-1.2883411718305444, -2.4434948198856086, -2.0563346358086174, -1.3185130145848347, -2.7683894322679112, 0.020524097286495235]
        self.poses =[  # Low camera poses
                    [-1.1764947592843438, -2.883647360768749, -2.432667823487778, -1.3184419912743557, -1.316452312525981, 0.020481730654609818],  
                    [-2.0624761579917683, -2.7427257741035858, -2.033343427633593, -2.2129389679422005, -2.673367740177965, -0.7062209396905752],
                    [-2.085183407148196, -2.842327709120595, -1.5037970953330635, -1.304918819318312, -3.592599725618643, 0.5700981978023888],
                    [-1.7922718849250963, -3.1786212242873604, -0.4493164078866475, -2.2150882380114867, -4.131287969822825, 0.2917182577430561],
                    [-1.4508166202808974, -3.375298717445945, -0.04562329227284502, -2.507577733357792, -4.5666512469176945, 0.09554792419923026],
                       # Medium camera poses
                    [-1.4408638525245268, -2.270611918990184, -2.3727612526037367, -2.512375469194263, -1.7054970972549208, -0.14235733835390985],
                    [-1.9421012341096429, -2.60487933315755, -1.682786812094579, -3.5087150889868397, -2.3640284246147276, -1.4721881013362497],
                    [-1.7682904090608664, -2.616438651776246, -1.366847624086397, -1.311583754919317, -4.228022830831442, 0.5828035183139911],
                    [-1.5666580045593337, -2.701830050829338, -1.0876324282923564, -1.5479961953068593, -4.463980522614415, 0.38582932131303366],
                       # High camera poses
                    [-1.2203962092554805, -2.122418234345, -1.909830426470517, -3.6303420959249757, -1.4712470328850218, 0.2974175140885674],
                    [-1.413329472624783, -2.2560878693111555, -1.5387472369052009, -4.089874237420497, -1.6194137098810062, -1.2970245992454874],
                    [-1.438694215664488, -2.309812088558476, -1.5104802980886154, -1.216825303519952, -4.666121972248885, 0.18324775341575922]]
        self.direction_map = {0: {'N': 5, 'S': 'illegal', 'E': 1, 'W': 'illegal'}, 
                              1: {'N': 6, 'S': 'illegal', 'E': 2, 'W': 0},
                              2: {'N': 7, 'S': 'illegal', 'E': 3, 'W': 1}, 
                              3: {'N': 8, 'S': 'illegal', 'E': 4, 'W': 2},
                              4: {'N': 8, 'S': 'illegal', 'E': 'illegal', 'W': 3}, 
                              5: {'N': 9, 'S': 0, 'E': 6, 'W': 'illegal'},
                              6: {'N': 10, 'S': 1, 'E': 7, 'W': 5}, 
                              7: {'N': 11, 'S': 2, 'E': 8, 'W': 6}, 
                              8: {'N': 11, 'S': 3, 'E': 'illegal', 'W': 7},
                              9: {'N': 'illegal', 'S': 5, 'E': 10, 'W': 'illegal'}, 
                              10: {'N': 'illegal', 'S': 6, 'E': 11, 'W': 9}, 
                              11: {'N': 'illegal', 'S': 7, 'E': 'illegal', 'W': 10}}

        _ = input("Go to start position? (y/n)\n")
        if _.lower() == 'y':
            self.robot.movej(self.poses[10], acc=self.accelaration, vel=self.velocity)
            self.current_pose = 10
        else:
            print("Robot not in known position")
            self.current_pose = None


    def init_object_detection(self, frozen_graph_path):
        # Read the graph.
        with tf.gfile.FastGFile(frozen_graph_path, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())


    def detect_image(self, img):

        if self.graph_def:
            with tf.Session() as sess:
                # Restore session
                sess.graph.as_default()
                tf.import_graph_def(self.graph_def, name='')

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

        else:
            print("Use init_object_detection() to load frozen graph first.")


    def close(self):
        self.robot.close()


    def go_to(self, pose_number):

        if -1 < pose_number < 12:
            if (self.current_pose == 0 and pose_number == 4) or (self.current_pose == 4 and pose_number == 0): 
                self.robot.movej(self.waypoint2, acc=self.accelaration, vel=self.velocity)

            elif (self.current_pose == 1 and pose_number == 4) or (self.current_pose == 4 and pose_number == 1):
                self.robot.movej(self.waypoint, acc=self.accelaration, vel=self.velocity)

            elif (self.current_pose == 5 and pose_number == 4) or (self.current_pose == 4 and pose_number == 5):
                self.robot.movej(self.waypoint2, acc=self.accelaration, vel=self.velocity)

            elif (self.current_pose == 8 and pose_number == 0) or (self.current_pose == 0 and pose_number == 8):
                self.robot.movej(self.waypoint2, acc=self.accelaration, vel=self.velocity)

            self.robot.movej(self.poses[pose_number], acc=self.accelaration, vel=self.velocity)
            self.current_pose = pose_number

        else:
            raise ValueError("Wrong pose number. Expected number from 0 to 11.")


    def go_to_random(self):
        pos = randint(0,11)
        self.go_to(pos)


    def go_direction(self, direction):
        """ Takes a direction (N, S, E, W) and moves the robot that way if possible
        """
        if direction.upper() == 'N' or direction.upper() == 'S' or direction.upper() == 'E' or direction.upper() == 'W':
            position = self.direction_map[self.current_pose][direction.upper()]
            if position == 'illegal':
                return 'illegal'
            else:
                self.go_to(position)
                return 'moved ' + direction.upper()

        else:
            raise ValueError("Directon not recognized.")



if __name__ == '__main__':
    new_robot = ProjectRobot()

    for _ in range(10):
        new_robot.go_to_random()
        print('Pose:', new_robot.current_pose)
    

    new_robot.close()