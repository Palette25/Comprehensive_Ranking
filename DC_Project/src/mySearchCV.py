# 2. max_depth
	bestMD = 3
	tempAccuracy = 0.0
	while params['max_depth'] <= 10:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: max_depth = %f, accuracy = %f' % (params['max_depth'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestMD = params['max_depth']
		params['max_depth'] += 1
	params['max_depth'] = bestMD

	print('Best Max_depth : %d' % bestMD)

	# 3. min_child_weight
	bestMCW = 1
	tempAccuracy = 0.0
	while params['min_child_weight'] <= 10:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: min_child_weight = %f, accuracy = %f' % (params['min_child_weight'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestMCW = params['min_child_weight']
		params['min_child_weight'] += 1
	params['min_child_weight'] = bestMCW

	print('Best Min_child_weight : %d' % bestMCW)

	# 4. gamma
	bestGamma = 0
	tempAccuracy = 0.0
	while params['gamma'] <= 1.0:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: gamma = %f, accuracy = %f' % (params['gamma'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestGamma = params['gamma']
		params['gamma'] += 0.1
	params['gamma'] = bestGamma

	print('Best Gamma : %f' % bestGamma)

	# 5. subsample
	bestSub = 0.5
	tempAccuracy = 0.0
	while params['subsample'] <= 1.0:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: subsample = %f, accuracy = %f' % (params['subsample'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestSub = params['subsample']
		params['subsample'] += 0.05
	params['subsample'] = bestSub

	print('Best subsample : %f' % bestSub)

	# 6. colsample_bytree
	bestCol = 0.5
	tempAccuracy = 0.0
	while params['colsample_bytree'] <= 1.0:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: colsample_bytree = %f, accuracy = %f' % (params['colsample_bytree'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestCol = params['colsample_bytree']
		params['colsample_bytree'] += 0.05
	params['colsample_bytree'] = bestCol

	print('Best Col : %f' % bestCol)

	# 7. alpha
	bestAlpha = 0.01
	tempAccuracy = 0.0
	while params['alpha'] <= 0.1:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: alpha = %f, accuracy = %f' % (params['alpha'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestAlpha = params['alpha']
		params['alpha'] += 0.01
	params['alpha'] = bestAlpha

	print('Best alpha : %f' % bestAlpha)

	# 8. lambda
	bestLambda = 0.01
	tempAccuracy = 0.0
	while params['lambda'] <= 0.1:
		stepAccuracy = xgbTrain(params, trainFeature, trainLabel, 500)
		print('XGB Train: lambda = %f, accuracy = %f' % (params['lambda'], stepAccuracy))
		if stepAccuracy > tempAccuracy:
			tempAccuracy = stepAccuracy
			bestLambda = params['lambda']
		params['lambda'] += 0.01
	params['lambda'] = bestLambda

	print('Best lambda : %f' % bestLambda)

	print(params)