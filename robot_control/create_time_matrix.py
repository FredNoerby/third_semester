import pickle
import time
import numpy as np
from project_robot import ProjectRobot 


# To load matrix use
# time_matrix = pickle.load(open("time_matrix.pkl", "rb" ))

rob = ProjectRobot()

time_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.int32)

converter_list = [['l', 0], ['l', 1], ['l', 2], ['l', 3], ['l', 4],
				  ['m', 0], ['m', 1], ['m', 2], ['m', 3],
				  ['h', 0], ['h', 1], ['h', 2]]

#counter = 0
for i in range(12):
	print("scoring for", converter_list[i][0], converter_list[i][1])
	rob.go_to(converter_list[i][0], converter_list[i][1])

	for j in range(12):
		start_time = int(round(time.time() * 1000))
		print("going to:", converter_list[j][0], converter_list[j][1])
		rob.go_to(converter_list[j][0], converter_list[j][1])
		time_matrix[i][j] = int(round(time.time() * 1000)) - start_time
		print(time_matrix[i][j], 'millis')
		print("returning to:", converter_list[i][0], converter_list[i][1])
		rob.go_to(converter_list[i][0], converter_list[i][1])
		#counter += 1

print(time_matrix)

#start_time = time.time()
#rob.go_to('l', 0)
#print(time.time() - start_time, 'seconds')


pickle.dump(time_matrix, open("time_matrix.pkl", "wb" ))
rob.close()

print("Times pickled, use following to load matrix:")
print('time_matrix = pickle.load(open("time_matrix.pkl", "rb" ))')