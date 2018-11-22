import numpy as np
import tensorflow as tf
import random
import cv2
import pickle
import time


class ProjectEnvironment:

    def __init__(self, simulated=True, ProjectRobot=None, pickled_bbox_dict='reinforcement_learning/bboxsGroundTruth.pkl', frozen_graph_path=None, banana_pose=0, print_log=True, video_cap=0, noise_sigma=30, save_detections=True):
        self.banana_pose = banana_pose
        self.reward_dict = {'move': -1, 'illegal': -5, 'guess_pos': 10, 'guess_neg': -10}
        self.observation_space = np.array([0, 0, 0, 0, 0])
        self.action_space = ActionSpace(12)
        self.history = []
        self.simulated = simulated
        self.print_log = print_log
        if not self.simulated:
            if not ProjectRobot or not frozen_graph_path:
                raise ValueError("Need ProjectRobot and frozen_graph_path in non-simulated environment.")
            else:
                self.robot = ProjectRobot
                self.save_detections = save_detections
                self.capture = cv2.VideoCapture(video_cap)
                self._init_object_detection(frozen_graph_path)
        else:
            self.sigma = noise_sigma
            self.cache = {0: None, 
                          1: None,
                          2: None, 
                          3: None,
                          4: None, 
                          5: None,
                          6: None, 
                          7: None, 
                          8: None,
                          9: None,
                          10: None,
                          11: None}
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
            
            with open(pickled_bbox_dict, 'rb') as handle: 
                self.simulated_bboxs = pickle.load(handle)
            

    def _init_object_detection(self, frozen_graph_path):
        """ Sets up object detection
        """
        # Read the graph.
        with tf.gfile.FastGFile(frozen_graph_path, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        # Warm up camera
        now = time.time()
        while now + 1 > time.time():
            ret, frame = self.capture.read()
            cv2.imshow('Capture', frame)
        cv2.destroyAllWindows()
        

    def detect_image(self):
        """ Uses the camera and an object detector to get a detection 
        """
        now = time.time()
        while now + 1 > time.time():
            ret, img = self.capture.read()
            cv2.imshow('Capture', img)
        cv2.destroyAllWindows()

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

                if self.save_detections:
                    # Save image with bounding box
                    cv2.imwrite('detections/b' + str(self.banana_pose) + time.strftime('_%d_%m_%H.%M.%S') + '.png',img)

                return img, detected

        else:
            print("Use init_object_detection() to load frozen graph first.")
    

    def _sense(self):
        """ Updates the bounding boxes 
        """
        if self.simulated:
            # If camera has already detected here reuse simulated detection
            if self.cache[self.observation_space[0]]:
                bbox = self.cache[self.observation_space[0]]
            # Else generate one
            else:
                mu = np.array(self.simulated_bboxs[self.banana_pose][self.observation_space[0]])
                x_offset = self.sigma * np.random.randn(1)
                y_offset = self.sigma * np.random.randn(1)
                bbox = [x_offset[0] + mu[0], y_offset[0] + mu[1], x_offset[0] + mu[2], y_offset[0] + mu[3]]
                self.cache[self.observation_space[0]] = bbox


            # Adds a bit of noise to the detection
            sigma = 2
            bbox = sigma * np.random.randn(1, 4) + bbox
            bbox = bbox.tolist()
            bbox = bbox[0]
            # print("Cache:")
            # print(self.cache)
            self.observation_space = [self.observation_space[0]] + bbox
            
            if self.print_log:
                print("[def _sense] Observation space:", self.observation_space)
        else:
            img, detected = self.detect_image()
            xmin, ymin = detected[0][2][0], detected[0][2][1]
            xmax, ymax = detected[0][2][2], detected[0][2][3]
            self.observation_space = [self.observation_space[0]] + [xmin, ymin, xmax, ymax]
            if self.print_log:
                print("[def _sense] Observation space:", self.observation_space)



    def _move(self, action):
        """ Moves robot or simulated position 
        """
        action_map = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}
        if self.simulated:
            move_to = self.direction_map[self.observation_space[0]][action_map[action]]
            if move_to == 'illegal':
                if self.print_log:
                    print("[def _move] Made illegal move:", action_map[action], "from", self.observation_space[0])
                return move_to
            else:
                if self.print_log:
                    print("[def _move] Made move:", action_map[action], "from", self.observation_space[0], "to", move_to)
                self.observation_space[0] = move_to
                return 'move'

        else:
            move_to = self.robot.go_direction(action_map[action])
            if move_to == 'illegal':
                if self.print_log:
                    print("[def _move] Made illegal move:", action_map[action], "from", self.observation_space[0])
                return move_to
            else:
                if self.print_log:
                    print("[def _move] Made move:", action_map[action], "from", self.observation_space[0], "to", move_to)
                self.observation_space[0] = move_to
                return 'move'



    def reset(self, banana_pose='rand', robot_position='rand'):
        """ Resets environment
        """
        if self.simulated:
            # Clear cache of bounding boxes
            self.cache = {0: None, 
                          1: None,
                          2: None, 
                          3: None,
                          4: None, 
                          5: None,
                          6: None, 
                          7: None, 
                          8: None,
                          9: None,
                          10: None,
                          11: None}
            if banana_pose == 'rand':
                self.banana_pose = random.randint(0, 7)
            elif -1 < banana_pose < 8:
                self.banana_pose = banana_pose
            else:
                raise ValueError("Banana pose has to be int from 0 to 7.")

            if robot_position == 'rand':
                self.observation_space[0] = random.randint(0, 11)
            else:
                self.observation_space[0] = robot_position
            self._sense()

        else:
            ba_p = int(input("**TYPE BANANA POSITION**\n"))
            self.banana_pose = ba_p
            self.observation_space[0] = self.robot.current_pose
            self._sense()
        if self.print_log:
            print("[def reset] Banana in position:", self.banana_pose)
        hist_dict = {"banana_pose": self.banana_pose, "observations": [self.observation_space], "actions": [], "rewards": [], "done": [], "banana_pred": None, "won": False}
        self.history.append(hist_dict)
        return self.observation_space


    def step(self, action):
        """ Takes a step based on an action
        """
        done = False
        if -1 < action < 4:
            rew = self._move(action)
            self._sense()
            reward = self.reward_dict[rew]
        elif 3 < action < 12:
            if action - 4 == self.banana_pose:
                if self.print_log:
                    print("[DONE] Made correct prediction: guessed", (action-4), ", banana position", self.banana_pose)
                reward = self.reward_dict['guess_pos']
                self.history[-1]['won'] = True

            else:
                if self.print_log:
                    print("[DONE] Made wrong prediction: guessed", (action-4), ", banana position", self.banana_pose)
                reward = self.reward_dict['guess_neg']
            self.history[-1]['banana_pred'] = action - 4
            done = True

        else:
            raise ValueError("Action must be from 0 to 11.")

        info_placeholder = {}

        # Stop the agent if it dosn't make a prediction in '_max_steps' steps
        _max_steps = 9
        if (len(self.history[-1]['actions']) > _max_steps) and (not done):
            done = True
            # Extra negative reward for not making a prediction
            reward = self.reward_dict['guess_neg']
            # Sets banana prediction to 8 to represent no prediction
            self.history[-1]['banana_pred'] = 8


        self.history[-1]['observations'].append(self.observation_space)
        self.history[-1]['actions'].append(action)
        self.history[-1]['rewards'].append(reward)
        self.history[-1]['done'].append(done)

        return self.observation_space, reward, done, info_placeholder


    def render(self, mode='human'):
        # Placeholder for rendering
        pass


    def close(self):
        if not self.simulated:
            self.capture.release()
            cv2.destroyAllWindows()
            self.robot.close()



class ActionSpace:
    def __init__(self, n):
        """ Initializes with the number of possible actions
        """
        self.n = n

    def sample(self):
        """ Returns a random sample from the action space
        """
        return random.randint(0, self.n-1)


if __name__ == "__main__":
    pass

