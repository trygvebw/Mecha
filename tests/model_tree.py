from sklearn import datasets

from mecha.models.model_tree import ModelTreeRegressor, ModelTreeForestRegressor

def model_tree_test():
	X, y = datasets.load_boston(return_X_y=True)
	mtr = ModelTreeRegressor().fit(X, y)
	mtr_pred = mtr.predict(X)
	assert len(mtr_pred.shape) == 1 and mtr_pred.shape[0] > 0

def model_tree_forest_test():
	X, y = datasets.load_boston(return_X_y=True)
	mtrf = ModelTreeForestRegressor().fit(X, y)
	mtrf_pred = mtr.predict(X)
	assert len(mtrf_pred.shape) == 1 and mtrf_pred.shape[0] > 0

if __name__ == '__main__':
	model_tree_test()
	model_tree_forest_test()