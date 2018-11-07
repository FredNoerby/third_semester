import numpy as np
import tensorflow as tf
import random
import cv2
import time


class ProjectEnvironment:

    def __init__(self, simulated=True, ProjectRobot=None, banana_pose=0, frozen_graph_path=None, print_log=True, video_cap=0):
        self.banana_pose = banana_pose
        self.reward_dict = {'move': -1, 'illegal': -5, 'guess_pos': 10, 'guess_neg': -10}
        self.move_dict = {}
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
                self.capture = cv2.VideoCapture(video_cap)
                self._init_object_detection(frozen_graph_path)
        else:
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
            
            self.simulated_bboxs = {0: {0: [155.88154792785645, 142.20349788665771, 525.6398773193359, 258.54174613952637], 
                                        1: [249.23639297485352, 177.40596771240234, 486.2113571166992, 287.5890254974365], 
                                        2: [260.3619956970215, 169.90003108978271, 421.49375915527344, 274.5183277130127], 
                                        3: [155.56610107421875, 158.94274234771729, 503.2339859008789, 283.9511775970459], 
                                        4: [48.01684856414795, 197.0718812942505, 529.6839141845703, 375.78349113464355], 
                                        5: [227.61255264282227, 59.99002933502197, 523.9230728149414, 227.96221733093262], 
                                        6: [297.80834197998047, 76.74675464630127, 459.04186248779297, 297.4820137023926], 
                                        7: [233.138427734375, 115.53388595581055, 551.078987121582, 311.9151020050049], 
                                        8: [118.80221366882324, 123.16254615783691, 479.1588592529297, 268.1375026702881], 
                                        9: [232.19457626342773, 84.66994285583496, 493.287353515625, 251.38461112976074], 
                                        10: [308.83962631225586, 119.49194669723511, 460.24280548095703, 372.0546340942383], 
                                        11: [194.73577499389648, 193.96461009979248, 453.07056427001953, 352.43579864501953]}, 
                                    1: {0: [95.72028160095215, 157.52986907958984, 453.1492614746094, 250.56747436523438], 
                                        1: [183.91090393066406, 186.4466142654419, 490.65818786621094, 259.2016410827637], 
                                        2: [216.18968963623047, 171.85064792633057, 470.61534881591797, 259.44711685180664], 
                                        3: [253.81879806518555, 160.66942691802979, 419.2533493041992, 292.1140193939209], 
                                        4: [79.84341144561768, 186.98474407196045, 534.0808868408203, 366.80614471435547], 
                                        5: [157.9542350769043, 94.78999614715576, 484.3600845336914, 218.2163143157959], 
                                        6: [229.9764060974121, 86.84665203094482, 536.0797882080078, 229.24861907958984], 
                                        7: [309.7332191467285, 89.67573165893555, 455.6769561767578, 334.9913692474365], 
                                        8: [217.6873016357422, 91.05239152908325, 453.29166412353516, 327.94535636901855], 
                                        9: [188.1252098083496, 106.78247451782227, 451.6790008544922, 283.1730365753174], 
                                        10: [270.63833236694336, 98.39431285858154, 459.8280715942383, 320.34247398376465], 
                                        11: [240.4990577697754, 161.60250663757324, 482.44022369384766, 343.34753036499023]}, 
                                    2: {0: [25.07568359375, 146.88777923583984, 347.791748046875, 280.2853488922119], 
                                        1: [116.97198867797852, 182.77586460113525, 413.29219818115234, 264.5684337615967], 
                                        2: [159.44199562072754, 172.87774085998535, 461.5235137939453, 240.57134628295898], 
                                        3: [240.05691528320312, 160.61214923858643, 482.28515625, 258.21189880371094], 
                                        4: [245.78460693359375, 180.1682996749878, 466.77791595458984, 379.1898536682129], 
                                        5: [126.3787841796875, 59.844582080841064, 392.7511978149414, 269.1928195953369], 
                                        6: [163.95868301391602, 104.39707517623901, 487.5782775878906, 210.58586597442627], 
                                        7: [272.0571708679199, 72.93819665908813, 535.8626937866211, 269.70170974731445], 
                                        8: [252.03828811645508, 62.42332935333252, 408.5616683959961, 304.79639053344727], 
                                        9: [204.01300430297852, 70.47919034957886, 380.54859161376953, 326.4375972747803], 
                                        10: [226.89939498901367, 127.676682472229, 477.47241973876953, 261.9780921936035], 
                                        11: [307.4538230895996, 122.05668926239014, 457.24254608154297, 370.17139434814453]}, 
                                    3: {0: [141.04486465454102, 138.46210956573486, 390.6951904296875, 332.4009418487549], 
                                        1: [151.16738319396973, 171.78926467895508, 352.1670150756836, 298.02294731140137], 
                                        2: [97.88689613342285, 168.80730628967285, 374.3408966064453, 257.2643852233887], 
                                        3: [169.9681282043457, 167.0174789428711, 476.76509857177734, 238.6279535293579], 
                                        4: [202.35370635986328, 185.7470941543579, 554.8125076293945, 320.25818824768066], 
                                        5: [216.9828224182129, 66.18666172027588, 398.05450439453125, 340.0555229187012], 
                                        6: [150.32119750976562, 68.0809736251831, 389.2893981933594, 269.36519622802734], 
                                        7: [197.0785903930664, 113.4481930732727, 519.0128326416016, 213.16821098327637], 
                                        8: [184.5335578918457, 72.96101331710815, 489.19666290283203, 214.42769050598145], 
                                        9: [213.31445693969727, 117.97780752182007, 415.5823516845703, 381.0872268676758], 
                                        10: [197.8969383239746, 104.86350059509277, 417.0574188232422, 297.9339122772217], 
                                        11: [270.66959381103516, 91.82139158248901, 482.31048583984375, 303.91940116882324]}, 
                                    4: {0: [74.28808689117432, 147.69158363342285, 486.8532180786133, 335.67174911499023], 
                                        1: [180.96012115478516, 177.99746990203857, 413.6406707763672, 324.8656940460205], 
                                        2: [125.39884567260742, 166.85779094696045, 339.05643463134766, 282.71830558776855], 
                                        3: [100.34422874450684, 157.01489925384521, 403.045654296875, 263.85074615478516], 
                                        4: [123.94803047180176, 197.58105754852295, 517.9644012451172, 289.2576313018799], 
                                        5: [159.85820770263672, 110.31777620315552, 475.63297271728516, 343.2647895812988], 
                                        6: [242.83538818359375, 78.57097148895264, 404.68189239501953, 339.6814727783203], 
                                        7: [143.83956909179688, 88.93903255462646, 437.7447509765625, 260.2907466888428], 
                                        8: [112.22294807434082, 99.35069561004639, 436.9428253173828, 218.70620727539062], 
                                        9: [197.7349090576172, 177.85114288330078, 462.91316986083984, 349.0167045593262], 
                                        10: [235.2197265625, 101.54769659042358, 375.07728576660156, 351.01407051086426], 
                                        11: [215.6536865234375, 121.86645984649658, 476.29234313964844, 251.74824714660645]}, 
                                    5: {0: [71.30443096160889, 154.00661945343018, 539.158821105957, 281.227970123291], 
                                        1: [126.4518928527832, 183.93908500671387, 484.9142837524414, 307.01019287109375], 
                                        2: [170.1566505432129, 167.26497173309326, 394.27661895751953, 295.3096389770508], 
                                        3: [122.19301223754883, 155.50676822662354, 346.6773986816406, 298.2521724700928], 
                                        4: [37.583560943603516, 181.4776611328125, 432.9759216308594, 323.9062213897705], 
                                        5: [160.9376335144043, 114.19156551361084, 525.9906387329102, 253.71265411376953], 
                                        6: [176.4317512512207, 113.41715097427368, 479.6759796142578, 340.3406810760498], 
                                        7: [201.60375595092773, 76.09294652938843, 400.14686584472656, 324.9565029144287], 
                                        8: [94.08977508544922, 68.91685724258423, 363.86226654052734, 279.62276458740234], 
                                        9: [233.33715438842773, 155.8577299118042, 505.0802993774414, 320.31386375427246], 
                                        10: [231.87152862548828, 146.33578777313232, 415.8656692504883, 385.484619140625], 
                                        11: [177.52361297607422, 129.53786373138428, 427.1985626220703, 292.09041595458984]}, 
                                    6: {0: [209.01090621948242, 145.82422256469727, 455.1850128173828, 309.13330078125], 
                                        1: [161.12546920776367, 180.74087619781494, 488.78185272216797, 289.0880012512207], 
                                        2: [94.1139030456543, 170.83820343017578, 456.52584075927734, 272.82139778137207], 
                                        3: [173.44049453735352, 160.48118591308594, 404.32044982910156, 305.7995796203613], 
                                        4: [69.23053741455078, 176.6786241531372, 364.5780563354492, 373.8046073913574], 
                                        5: [251.90349578857422, 82.73290872573853, 485.0423049926758, 321.8799018859863], 
                                        6: [151.73564910888672, 116.09292268753052, 520.6997680664062, 250.5948543548584], 
                                        7: [206.16409301757812, 97.03665018081665, 468.8762664794922, 344.8062801361084], 
                                        8: [207.2402000427246, 60.27714014053345, 363.2143020629883, 335.8345699310303], 
                                        9: [301.6863441467285, 106.27241134643555, 458.5652542114258, 378.19790840148926], 
                                        10: [209.37143325805664, 196.1799144744873, 449.8310089111328, 324.3376922607422], 
                                        11: [230.3848648071289, 90.74046850204468, 380.0343704223633, 339.47364807128906]}, 
                                    7: {0: [239.76892471313477, 142.21079349517822, 496.6939926147461, 304.7563076019287], 
                                        1: [266.1703109741211, 181.53059005737305, 449.14161682128906, 306.01195335388184], 
                                        2: [142.1974277496338, 170.92527866363525, 477.60406494140625, 271.6724967956543], 
                                        3: [100.29245376586914, 169.41555976867676, 455.48248291015625, 287.51235008239746], 
                                        4: [139.82773780822754, 186.08678340911865, 400.2907943725586, 412.1627426147461], 
                                        5: [291.742000579834, 65.59603214263916, 448.9176559448242, 313.72246742248535], 
                                        6: [253.24373245239258, 103.82404088973999, 516.0369110107422, 323.0983543395996], 
                                        7: [148.74591827392578, 133.76572608947754, 514.5703125, 314.1383457183838], 
                                        8: [124.46242332458496, 104.32380437850952, 404.525146484375, 341.33846282958984], 
                                        9: [297.5815773010254, 74.47177648544312, 457.45269775390625, 334.687557220459], 
                                        10: [246.6046905517578, 183.76637935638428, 481.27880096435547, 345.6130599975586], 
                                        11: [223.76007080078125, 126.72177314758301, 396.4906311035156, 378.724422454834]}}

    def _init_object_detection(self, frozen_graph_path):
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
                return img, detected

        else:
            print("Use init_object_detection() to load frozen graph first.")
    

    def _sense(self):
        if self.simulated:
            bbox = np.array(self.simulated_bboxs[self.banana_pose][self.observation_space[0]])
            bbox = bbox + np.random.randn(1, 4)
            bbox = bbox.tolist()
            bbox = bbox[0]
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
        if self.simulated:
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
            """if banana_pose == 'rand' or -1 > banana_pose or banana_pose > 8:
                raise ValueError("Need correct banana pose in non-simulated environment.")
            else:
                self.banana_pose = banana_pose
                self.observation_space[0] = self.robot.current_pose
                self._sense()"""
            ba_p = int(input("**TYPE BANANA POSITION**\n"))
            self.banana_pose = ba_p
            self.observation_space[0] = self.robot.current_pose
            self._sense()
        if self.print_log:
            print("[def reset] Banana in position:", self.banana_pose)
        hist_dict = {"banana_pose": self.banana_pose, "observations": [self.observation_space], "actions": [], "rewards": [], "done": []}
        self.history.append(hist_dict)
        return self.observation_space


    def step(self, action):
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
            else:
                if self.print_log:
                    print("[DONE] Made wrong prediction: guessed", (action-4), ", banana position", self.banana_pose)
                reward = self.reward_dict['guess_neg']
            done = True

        else:
            raise ValueError("Action must be from 0 to 11.")

        info_placeholder = {}
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

