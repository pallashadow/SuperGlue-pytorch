import torch

def superglueLoss(pred, data):
	epsilon = 1e-6
	all_matches = data['all_matches']# shape=torch.Size([1, 87, 2])
	nMatch = data['num_match_list']
	scores = pred['scores'].exp() # shape=(1,N,M)
	#print(torch.sum(scores, dim=1)) #check scores is normalized, out: [1,1,1,1,...,N]
	''' # slow
	loss = []
	for batchIdx, all_match in enumerate(all_matches):
		nm = int(nMatch[batchIdx][0])
		loss += [-torch.log(scores[batchIdx,x,y]+epsilon) for x, y in all_match[:nm]]
	loss_mean = torch.mean(torch.stack(loss))
	'''
	indices = [[batchIdx, x, y] for batchIdx, all_match in enumerate(all_matches) for x, y in all_match[:int(nMatch[batchIdx][0])]]
	indices = torch.LongTensor(indices).t().chunk(chunks=3, dim=0) # batch index selection
	loss = -torch.log(scores[indices]+epsilon)
	loss_mean = torch.mean(loss).reshape(1, -1)
	return loss_mean
