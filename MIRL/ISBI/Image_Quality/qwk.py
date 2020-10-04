
from sklearn.metrics import confusion_matrix
class QWKEvaluation(Callback):
	def __init__(self, valid_data = valid_ls, path = 'data/',
		img_size = (512,512), num_class= 5,interval=1):
		super(Callback, self).__init__()
		self.valid_data = valid_ls
		self.n_classes = num_class
		self.interval = interval
		self.path = path
		self.sigmaX =10
		self.img_size = img_size
		self.history = []
		# self.y_val = to_categorical(y_or, num_classes=self.n_classes)

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred_o = []
			y_true_o = []
			for i, file in enumerate(self.valid_data):
				jpgfile = cv2.imread(self.path + file[0] + '.jpg')
				# jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_BGR2RGB)
				if(jpgfile.shape != (512,512,3)):
					jpgfile = cv2.resize(jpgfile, self.img_size)
				jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
				# jpgfile = Image.open(self.path + file[0] + '.jpeg')
				# jpgfile = jpgfile.resize(self.img_size, Image.ANTIALIAS)
				jpgfile = np.expand_dims(jpgfile , axis=0)
				jpgfile = np.array(jpgfile, np.float32) / 255
				y_pred_o.append(model.predict(jpgfile, steps=None))
				y_true_o.append(int(file[1]/2))

			y_pred_o = np.squeeze(np.array(y_pred_o))
			y_true_o = np.array(y_true_o)


			def flatten(y):
				return np.argmax(y, axis=1).reshape(-1)

			print("y_true_o................",y_true_o[:100])
			print("y_pred_o................", flatten(y_pred_o)[:100])
			score = cohen_kappa_score(y_true_o, flatten(y_pred_o),
				weights='quadratic')
			print("cohen kappa",score)

			print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
			self.history.append(score)
			if score >= max(self.history):
				print('saving checkpoint: ', score)
				# self.model.save('../working/densenet_bestqwk.h5')
				self.model.save_weights('a01_' + str(epoch+1) +'_file.h5')
				confus_matrix = confusion_matrix(y_true_o, flatten(y_pred_o))
				print("Confusion matrix:\n%s"% confus_matrix)

# valid_ls_qwk = valid_ls[:int(0.4*len(valid_ls))] + valid_ls[int(0.6*len(valid_ls)):]
qwk = QWKEvaluation(valid_data=valid_ls, interval=1, img_size = IMG_SIZE)

