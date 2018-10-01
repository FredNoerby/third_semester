import urx
from random import randint

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
        self.waypoint = [-1.6616261926776854, -2.516756425648164, -1.7162176543532885, -2.035364287737379, -3.275768284901913, 0.021001491282445117]
        self.waypoint2 = [-1.2883411718305444, -2.4434948198856086, -2.0563346358086174, -1.3185130145848347, -2.7683894322679112, 0.020524097286495235]
        self.poses =[[
                    [-1.1764947592843438, -2.883647360768749, -2.432667823487778, -1.3184419912743557, -1.316452312525981, 0.020481730654609818],
                    [-2.0624761579917683, -2.7427257741035858, -2.033343427633593, -2.2129389679422005, -2.673367740177965, -0.7062209396905752],
                    [-2.085183407148196, -2.842327709120595, -1.5037970953330635, -1.304918819318312, -3.592599725618643, 0.5700981978023888],
                    [-1.7922718849250963, -3.1786212242873604, -0.4493164078866475, -2.2150882380114867, -4.131287969822825, 0.2917182577430561],
                    [-1.4508166202808974, -3.375298717445945, -0.04562329227284502, -2.507577733357792, -4.5666512469176945, 0.09554792419923026]],
                    [
                    [-1.4408638525245268, -2.270611918990184, -2.3727612526037367, -2.512375469194263, -1.7054970972549208, -0.14235733835390985],
                    [-1.9421012341096429, -2.60487933315755, -1.682786812094579, -3.5087150889868397, -2.3640284246147276, -1.4721881013362497],                    [-1.7682904090608664, -2.616438651776246, -1.366847624086397, -1.311583754919317, -4.228022830831442, 0.5828035183139911],
                    [-1.5666580045593337, -2.701830050829338, -1.0876324282923564, -1.5479961953068593, -4.463980522614415, 0.38582932131303366]],
                    [
                    [-1.2203962092554805, -2.122418234345, -1.909830426470517, -3.6303420959249757, -1.4712470328850218, 0.2974175140885674],
                    [-1.413329472624783, -2.2560878693111555, -1.5387472369052009, -4.089874237420497, -1.6194137098810062, -1.2970245992454874],
                    [-1.438694215664488, -2.309812088558476, -1.5104802980886154, -1.216825303519952, -4.666121972248885, 0.18324775341575922]]]
        _ = input("Go to start position? (y/n)\n")
        if _.lower() == 'y':
            self.robot.movej(self.poses[2][1], acc=self.accelaration, vel=self.velocity)
            self.current_height = 'h'
            self.current_spot = 1
        else:
            print("Robot not in known position")
            self.current_height = None
            self.current_spot = None


    def close(self):
        self.robot.close()


    def go_to(self, height, spot):
        if self.current_height == None or self.current_spot == None:
            print("Robot position unknown. Going to start position.")
            self.robot.movej(self.poses[2][1], acc=self.accelaration, vel=self.velocity)
            self.current_height = 'h'
            self.current_spot = 1

        if height.lower() == 'l' or height.lower() == 'low':
            if -1 < spot < 5:
                if self.current_height == 'l':
                    if abs(self.current_spot - spot) > 2:
                        if (self.current_spot == 0 and spot == 4) or (self.current_spot == 4 and spot == 0): 
                            self.robot.movej(self.waypoint2, acc=self.accelaration, vel=self.velocity)
                        else:
                            self.robot.movej(self.waypoint, acc=self.accelaration, vel=self.velocity)

                elif self.current_height == 'm':
                    if (self.current_spot == 0 and spot == 4) or (self.current_spot == 3 and spot == 0): 
                            self.robot.movej(self.waypoint2, acc=self.accelaration, vel=self.velocity)

                self.robot.movej(self.poses[0][spot], acc=self.accelaration, vel=self.velocity)
                self.current_height = 'l'
                self.current_spot = spot

            else:
                raise ValueError("If height is low, spot must be 0, 1, 2, 3, or 4")

        elif height.lower() == 'm' or height.lower() == 'mid':
            if -1 < spot < 4:
                if (self.current_height == 'l' and self.current_spot == 0 and spot == 4) or (self.current_height == 'l' and self.current_spot == 4 and spot == 0):
                    self.robot.movej(self.waypoint2, acc=self.accelaration, vel=self.velocity)
                self.robot.movej(self.poses[1][spot], acc=self.accelaration, vel=self.velocity)
                self.current_height = 'm'
                self.current_spot = spot
            else:
                raise ValueError("If height is mid, spot must be 0, 1, 2, or 3")

        elif height.lower() == 'h' or height.lower() == 'high':
            if -1 < spot < 3:
                self.robot.movej(self.poses[2][spot], acc=self.accelaration, vel=self.velocity)
                self.current_height = 'h'
                self.current_spot = spot
            else:
                raise ValueError("If height is high, spot must be 0, 1, or 2")

        else:
            raise ValueError("Wrong height. Expected 'l', 'm', 'h', 'low', 'mid' or 'high'")


    def go_to_random(self):
        height = randint(0,2)
        if height == 0:
            height = 'l'
            spot = randint(0,4)
        elif height == 1:
            height = 'm'
            spot = randint(0,3)
        else:
            height = 'h'
            spot = randint(0,2)
        self.go_to(height, spot)



if __name__ == '__main__':
    new_robot = ProjectRobot()

    for _ in range(100):
        new_robot.go_to_random()
        print('Height:', new_robot.current_height, 'Spot:', new_robot.current_spot)
    

    new_robot.close()