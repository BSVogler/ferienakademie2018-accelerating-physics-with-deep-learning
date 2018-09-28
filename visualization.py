import numpy as np
import matplotlib.pyplot as plt

__all__ = ['vis']

"""
A tool for visualization the results:

Parameters
----------
prediction: The predicted air-flow distribution, 3d numpy array, typical shape is (64,64,3)
truth	  : The ground truth of air-flow distribution, the same shape as predicition

Returns
----------
Nothing, but pop up a window that shows visualized results
"""

def vis(prediction, truth):
	plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
	# predicted data
	plt.subplot(331)
	plt.title('Predicted pressure')
	plt.imshow(prediction[:,:,0], cmap='jet')# vmin=-100,vmax=100, cmap='jet')
	plt.colorbar()
	plt.subplot(332)
	plt.title('Predicted x velocity')
	plt.imshow(prediction[:,:,1], cmap='jet')
	plt.colorbar()
	plt.subplot(333)
	plt.title('Predicted y velocity')
	plt.imshow(prediction[:,:,2], cmap='jet')
	plt.colorbar()

	# groundtruth data
	plt.subplot(334)
	plt.title('Ground truth pressure')
	plt.imshow(truth[:,:,0],cmap='jet')
	plt.colorbar()
	plt.subplot(335)
	plt.title('Ground truth x velocity')
	plt.imshow(truth[:,:,1],cmap='jet')
	plt.colorbar()
	plt.subplot(336)
	plt.title('Ground truth y velocity')
	plt.imshow(truth[:,:,2],cmap='jet')
	plt.colorbar()

	# difference
	plt.subplot(337)
	plt.title('Difference pressure')
	plt.imshow((truth[:,:,0] - prediction[:,:,0]),cmap='jet')
	plt.colorbar()
	plt.subplot(338)
	plt.title('Difference x velocity')
	plt.imshow((truth[:,:,1] - prediction[:,:,0]),cmap='jet')
	plt.colorbar()
	plt.subplot(339)
	plt.title('Difference y velocity')
	plt.imshow((truth[:,:,1] - prediction[:,:,2]),cmap='jet')
	plt.colorbar()

	plt.show()