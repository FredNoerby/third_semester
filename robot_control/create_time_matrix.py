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

#counter = 0
for i in range(12):
	print("scoring for: position", str(i))
	rob.go_to(i)

	for j in range(12):
		start_time = int(round(time.time() * 1000))
		print("going to: position", str(j))
		rob.go_to(j)
		time_matrix[i][j] = int(round(time.time() * 1000)) - start_time
		print(time_matrix[i][j], 'millis')
		print("returning to: position", str(i))
		rob.go_to(i)
		#counter += 1

print(time_matrix)

#start_time = time.time()
#rob.go_to('l', 0)
#print(time.time() - start_time, 'seconds')


pickle.dump(time_matrix, open("time_matrix.pkl", "wb" ))
rob.close()

print("Times pickled, use following to load matrix:")
print('time_matrix = pickle.load(open("time_matrix.pkl", "rb" ))')