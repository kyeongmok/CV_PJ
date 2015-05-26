#include <math.h>
#include <random>
#include <fstream>

#include "DecisionTree.h"
#include "LBF.h"

//basic constructor
DecisionTree::DecisionTree()
	: binLength(0)
	, pRootNode(nullptr)
{
	pRootNode = new Node();
	pRootNode->pParent	= pRootNode;
	pRootNode->isLeaf	= true;
	pRootNode->depth	= 1;
}

//copy constructor
DecisionTree::DecisionTree(const DecisionTree &Other)
{
	//maxDepth = Other.maxDepth;
	binLength = Other.binLength;

	pRootNode = new Node();
	pRootNode->pParent = pRootNode;
	pRootNode->isLeaf = true;
	pRootNode->depth = 1;
	pRootNode->sampleIdxs = Other.pRootNode->sampleIdxs;
	pRootNode->threshold = Other.pRootNode->threshold;
	pRootNode->featPair = Other.pRootNode->featPair;

	std::vector<Node*> srcCurRootNodes;
	srcCurRootNodes.push_back(Other.pRootNode);
	std::vector<Node*> dstCurRootNodes;
	dstCurRootNodes.push_back(pRootNode);

	while (1)
	{
		std::vector<Node*> srcNextRootNodes;
		std::vector<Node*> dstNextRootNodes;
		for (int j = 0; j < srcCurRootNodes.size(); j++)
		{
			if (srcCurRootNodes[j]->plChild)
			{
				dstCurRootNodes[j]->plChild = new Node();
				dstCurRootNodes[j]->plChild->pParent = dstCurRootNodes[j];
				dstCurRootNodes[j]->plChild->isLeaf = true;
				dstCurRootNodes[j]->isLeaf = false;
				dstCurRootNodes[j]->plChild->depth = dstCurRootNodes[j]->depth + 1;
				dstCurRootNodes[j]->plChild->sampleIdxs = srcCurRootNodes[j]->plChild->sampleIdxs;
				dstCurRootNodes[j]->plChild->threshold = srcCurRootNodes[j]->plChild->threshold;
				dstCurRootNodes[j]->plChild->featPair = srcCurRootNodes[j]->plChild->featPair;

				srcNextRootNodes.push_back(srcCurRootNodes[j]->plChild);
				dstNextRootNodes.push_back(dstCurRootNodes[j]->plChild);
			}
			if (srcCurRootNodes[j]->prChild)
			{
				dstCurRootNodes[j]->prChild = new Node();
				dstCurRootNodes[j]->prChild->pParent = dstCurRootNodes[j];
				dstCurRootNodes[j]->prChild->isLeaf = true;
				dstCurRootNodes[j]->isLeaf = false;
				dstCurRootNodes[j]->prChild->depth = dstCurRootNodes[j]->depth + 1;
				dstCurRootNodes[j]->prChild->sampleIdxs = srcCurRootNodes[j]->prChild->sampleIdxs;
				dstCurRootNodes[j]->prChild->threshold = srcCurRootNodes[j]->prChild->threshold;
				dstCurRootNodes[j]->prChild->featPair = srcCurRootNodes[j]->prChild->featPair;

				srcNextRootNodes.push_back(srcCurRootNodes[j]->prChild);
				dstNextRootNodes.push_back(dstCurRootNodes[j]->prChild);
			}
		}
		if (srcNextRootNodes.empty())
			break;
		srcCurRootNodes.clear();
		srcCurRootNodes = srcNextRootNodes;
		dstCurRootNodes.clear();
		dstCurRootNodes = dstNextRootNodes;
	}
	vpLeafNodes.clear();
	FindLeafNodes(pRootNode);
}

//destructor
DecisionTree::~DecisionTree()
{
	vpLeafNodes.clear();
	Destroyer(pRootNode);
	pRootNode = nullptr;
}

//function for destructing nodes recursively
void DecisionTree::Destroyer(Node* pSubRootNode)
{
	if (IsLeafNode(pSubRootNode))
	{
		delete pSubRootNode;
		pSubRootNode = nullptr;
		return;
	}
	else
	{
		if (pSubRootNode->plChild)
		{
			Destroyer(pSubRootNode->plChild);
		}
		if (pSubRootNode->prChild)
		{
			Destroyer(pSubRootNode->prChild);
		}
		delete pSubRootNode;
		pSubRootNode = nullptr;
	}
}

//function for finding leaf nodes recursively
void DecisionTree::FindLeafNodes(Node* pSubRootNode)
{
	/* first caller must do clearing vpLeafNodes vector */
	if (IsLeafNode(pSubRootNode))
	{
		vpLeafNodes.push_back(pSubRootNode);
		return;
	}
	else
	{
		if (pSubRootNode->plChild)
		{
			FindLeafNodes(pSubRootNode->plChild);
		}
		if (pSubRootNode->prChild)
		{
			FindLeafNodes(pSubRootNode->prChild);
		}
	}
}

//function for saving data recursively
void DecisionTree::Saver(Node* pSubRootNode, std::ofstream& outStream)
{
	if (!pSubRootNode)
	{
		outStream << "X " << std::endl;	//sign for null node
		return;
	}
	else
	{
		outStream << "O ";	//sign for valid node
		outStream << pSubRootNode->isLeaf << " " << pSubRootNode->depth << " " << pSubRootNode->threshold << " " << pSubRootNode->featPair.first.x << " " << pSubRootNode->featPair.first.y << " ";
		outStream << pSubRootNode->featPair.second.x << " " << pSubRootNode->featPair.second.y << std::endl;
		Saver(pSubRootNode->plChild, outStream);
		Saver(pSubRootNode->prChild, outStream);
	}
}

//function for loading data recursively
void DecisionTree::Loader(Node* pSubRootNode, std::ifstream& inStream)
{
	inStream >> pSubRootNode->isLeaf >> pSubRootNode->depth >> pSubRootNode->threshold >> pSubRootNode->featPair.first.x >> pSubRootNode->featPair.first.y;
	inStream >> pSubRootNode->featPair.second.x >> pSubRootNode->featPair.second.y;
	
	std::string labelStr;
	inStream >> labelStr;
	if (labelStr == "X")
	{
		inStream >> labelStr;
		assert(labelStr == "X"); //if one of child is null, then another child also should be null
		return;
	}
	else
	{
		pSubRootNode->plChild = new Node();
		pSubRootNode->plChild->pParent = pSubRootNode;
		Loader(pSubRootNode->plChild, inStream);

		inStream >> labelStr;
		assert(labelStr == "O"); //if one of child is valid, then another child also should be valid
		pSubRootNode->prChild = new Node();
		pSubRootNode->prChild->pParent = pSubRootNode;
		Loader(pSubRootNode->prChild, inStream);
	}

}

int DecisionTree::GetTreeDepth()
{
	if (vpLeafNodes.empty())
	{
		FindLeafNodes(pRootNode);
	}
	int depth = 0;
	for (auto v : vpLeafNodes)
	{
		depth = (v->depth > depth) ? v->depth : depth;
	}
	return depth;
}

void DecisionTree::Clear()
{
	Destroyer(pRootNode);
	vpLeafNodes.clear();
	binLength = 0;

	pRootNode = new Node();
	pRootNode->pParent = pRootNode;
	pRootNode->isLeaf = true;
	pRootNode->depth = 1;
}

void DecisionTree::SplitNode(Node* pNode, const std::vector<Data>& vData, const Params& params, int stage, int lmarkIdx)
{
	/* split the parent node */
	pNode->isLeaf = false;

	pNode->plChild = new Node();
	Node& lChild = *(pNode->plChild);
	lChild.pParent = pNode;
	lChild.isLeaf = true;
	lChild.depth = pNode->depth + 1;

	pNode->prChild = new Node();
	Node& rChild = *(pNode->prChild);
	rChild.pParent = pNode;
	rChild.isLeaf = true;
	rChild.depth = pNode->depth + 1;

	/* choose the best pair of features based on variance reduction */
	//generate candidate features
	int numCandidates = params.maxNumFeatures[stage];
	int numSamples = pNode->sampleIdxs.size();
	std::vector<std::pair<LBF_DATA, LBF_DATA>> radiusPair, anglePair;
	GetProposals(numCandidates, params.radius, params.angles, radiusPair, anglePair);	//random procedure

	std::vector<std::pair<LBF_DATA, LBF_DATA>> anglePairCos(numCandidates), anglePairSin(numCandidates);
	for (int i = 0; i < numCandidates; i++)
	{
		anglePairCos[i].first = cv::cos(anglePair[i].first);
		anglePairCos[i].second = cv::cos(anglePair[i].second);
		anglePairSin[i].first = cv::sin(anglePair[i].first);
		anglePairSin[i].second = cv::sin(anglePair[i].second);
	}

	//extract pixel difference from pairs
	std::vector<std::vector<int>> pixelDiffs(numCandidates);
	for (int i = 0; i < numCandidates; i++)
	{
		pixelDiffs[i].resize(numSamples);
	}

	Shape shapeResidual(numSamples);
	for (int i = 0; i < numSamples; i++)
	{
		int dataIdx = std::floor((float)pNode->sampleIdxs[i] / params.augnumber);
		int augIdx = pNode->sampleIdxs[i] % params.augnumber;

		//calculate the relative location under the coordinate of meanShape
		Shape pixelCandidatesA(numCandidates), pixelCandidatesB(numCandidates);
		for (int j = 0; j < numCandidates; j++)
		{
			pixelCandidatesA.data()[j].x = anglePairCos[j].first * radiusPair[j].first * params.maxRatioRadius[stage] * vData[dataIdx].intermediateBboxes[stage][augIdx].width;
			pixelCandidatesA.data()[j].y = anglePairSin[j].first * radiusPair[j].first * params.maxRatioRadius[stage] * vData[dataIdx].intermediateBboxes[stage][augIdx].height;
			pixelCandidatesB.data()[j].x = anglePairCos[j].second * radiusPair[j].second * params.maxRatioRadius[stage] * vData[dataIdx].intermediateBboxes[stage][augIdx].width;
			pixelCandidatesB.data()[j].y = anglePairSin[j].second * radiusPair[j].second * params.maxRatioRadius[stage] * vData[dataIdx].intermediateBboxes[stage][augIdx].height;
		}

		//transform the pixels from coordinate of meanShape to coordinate of current shape
		cv::Mat matA, matB;
		pixelCandidatesA.ToTransMat(matA);
		pixelCandidatesB.ToTransMat(matB);

		
		LBF::DoGeometricTransform(vData[dataIdx].meanShape2tf[augIdx], matA, matA);
		LBF::DoGeometricTransform(vData[dataIdx].meanShape2tf[augIdx], matB, matB);

		pixelCandidatesA.FromTransMat(matA);
		pixelCandidatesB.FromTransMat(matB);

		for (int j = 0; j < numCandidates; j++)
		{
			pixelCandidatesA.data()[j] += vData[dataIdx].intermediateShapes[stage][augIdx].at(lmarkIdx);
			pixelCandidatesB.data()[j] += vData[dataIdx].intermediateShapes[stage][augIdx].at(lmarkIdx);

			pixelCandidatesA.data()[j].x = std::max(0, std::min((int)(pixelCandidatesA.data()[j].x + 0.5), vData[dataIdx].curWidth - 1));	//need interpolation
			pixelCandidatesA.data()[j].y = std::max(0, std::min((int)(pixelCandidatesA.data()[j].y + 0.5), vData[dataIdx].curHeight - 1));	//need interpolation
			pixelCandidatesB.data()[j].x = std::max(0, std::min((int)(pixelCandidatesB.data()[j].x + 0.5), vData[dataIdx].curWidth - 1));	//need interpolation
			pixelCandidatesB.data()[j].y = std::max(0, std::min((int)(pixelCandidatesB.data()[j].y + 0.5), vData[dataIdx].curHeight - 1));	//need interpolation

		}

		//calculate pixel difference
		for (int j = 0; j < numCandidates; j++)
		{
			pixelDiffs[j][i] = (int)vData[dataIdx].srcImgGray->at<uchar>((int)pixelCandidatesA.data()[j].y, (int)pixelCandidatesA.data()[j].x) -
							   (int)vData[dataIdx].srcImgGray->at<uchar>((int)pixelCandidatesB.data()[j].y, (int)pixelCandidatesB.data()[j].x);
		}

		shapeResidual.data()[i] = vData[dataIdx].shapeResidual[augIdx].at(lmarkIdx);
	}

	//calculate variances
	LBF_POINT E = shapeResidual.GetMeanPoint();
	LBF_POINT E2 = shapeResidual.GetSquareMeanPoint();
	LBF_DATA entireVariance = numSamples * ((E2.x - E.x * E.x) + (E2.y - E.y * E.y));

	//select pairs based on variance reduction
	int maxStep = 1;	//incomplete code
	std::vector<std::vector<LBF_DATA>> varReductions(numCandidates);
	std::vector<std::vector<int>> thresholds(numCandidates);
	std::vector<std::vector<int>> pixelDiffSorted(pixelDiffs.size());
	for (int i = 0; i < numCandidates; i++)
	{
		pixelDiffSorted[i].assign(pixelDiffs[i].begin(), pixelDiffs[i].end());
		std::sort(pixelDiffSorted[i].begin(), pixelDiffSorted[i].end());
	}

	std::random_device randDev;
	std::mt19937 generator(randDev());
	std::uniform_real_distribution<LBF_DATA> distr(0, 1);
	for (int i = 0; i < numCandidates; i++)
	{
		// for(int curStep = 0; curStep < maxStep; curStep++)
		int curStep = 0;	//incomplete code

		int idx = std::ceil(numSamples * (0.5 + 0.9 * (distr(generator) - 0.5))) - 1;	//random procedure
		int threshold = pixelDiffSorted[i][idx];
		thresholds[i].resize(maxStep);
		thresholds[i][curStep] = threshold;
		std::vector<int> idxLC, idxRC;
		for (int j = 0; j < numSamples; j++)
		{
			if (pixelDiffs[i][j] < threshold)
				idxLC.push_back(j);
			else
				idxRC.push_back(j);
		}

		LBF_POINT lcE, rcE, lcE2, rcE2;
		lcE = shapeResidual.GetMeanUsedPoint(idxLC);
		rcE = shapeResidual.GetMeanUsedPoint(idxRC);
		lcE2 = shapeResidual.GetSquareMeanUsedPoint(idxLC);
		rcE2 = shapeResidual.GetSquareMeanUsedPoint(idxRC);

		LBF_DATA lcVariance = (lcE2.x + lcE2.y) - (lcE.x * lcE.x + lcE.y * lcE.y);
		LBF_DATA rcVariance = (rcE2.x + rcE2.y) - (rcE.x * rcE.x + rcE.y * rcE.y);

		LBF_DATA varReduce = entireVariance - idxLC.size() * lcVariance - idxRC.size() * rcVariance;

		varReductions[i].resize(maxStep);
		varReductions[i][curStep] = varReduce;
	}

	int colMaxIdx = 0;
	int stepMaxIdx = 0;	//incomplete code
	LBF_DATA maxReduction = varReductions[0][0];
	for (int i = 1; i < varReductions.size(); i++)
	{
		if (maxReduction < varReductions[i][0])
		{
			maxReduction = varReductions[i][0];
			colMaxIdx = i;
		}
	}
	//isValid 처리 필요 : maxReduction <= 0 일 때, 즉 variance reduction이 더 이상 되지 않을 때 어떻게 처리할 지
	// 1. 일단 위쪽부터 random procedure를 다시 수행해서 양수의 variance reduction이 나올 때까지 반복
	// 2. 1을 해도 variance reduction이 되지 않을 경우(음수만 나올 경우) => 이론적으로는 split을 하지 않는 것이 맞음. 어떻게 처리할 지?

	/* write the trained feature pair, threshold, sampleIdxs to Nodes */
	pNode->threshold = thresholds[colMaxIdx][stepMaxIdx];
	pNode->featPair.first = { anglePair[colMaxIdx].first, radiusPair[colMaxIdx].first };
	pNode->featPair.second = { anglePair[colMaxIdx].second, radiusPair[colMaxIdx].second };

	for (int i = 0; i < numSamples; i++)
	{
		if (pixelDiffs[colMaxIdx][i] < pNode->threshold)
			lChild.sampleIdxs.push_back(pNode->sampleIdxs[i]);
		else
			rChild.sampleIdxs.push_back(pNode->sampleIdxs[i]);
	}

	return;
}

void DecisionTree::GetProposals(int numProposals, const std::vector<LBF_DATA>& radiusGrid, const std::vector<LBF_DATA>& anglesGrid,
	std::vector<std::pair<LBF_DATA, LBF_DATA>>& radiusPairs, std::vector<std::pair<LBF_DATA, LBF_DATA>>& anglePairs)
{
	std::random_device randDev;
	std::mt19937 generator(randDev());

	std::vector<int> randPermA(radiusGrid.size() * anglesGrid.size(), 0);
	std::vector<int> randPermB(radiusGrid.size() * anglesGrid.size(), 0);
	for (int i = 0; i < randPermA.size(); i++)
	{
		randPermA[i] = randPermB[i] = i;
	}

	std::shuffle(randPermA.begin(), randPermA.end(), generator);
	std::shuffle(randPermB.begin(), randPermB.end(), generator);

	int vecSize = randPermA.size();
	for (int i = 0; i < vecSize; i++)
	{
		if (randPermA[i] == randPermB[i])
		{
			randPermA.erase(randPermA.begin() + i);
			randPermB.erase(randPermB.begin() + i);
			i--;
			vecSize--;
		}
	}

	assert(randPermA.size() > numProposals);

	randPermA.erase(randPermA.begin() + numProposals, randPermA.end());
	randPermB.erase(randPermB.begin() + numProposals, randPermB.end());

	radiusPairs.resize(randPermA.size());
	anglePairs.resize(randPermA.size());
	for (int i = 0; i < randPermA.size(); i++)
	{
		int radiusIdxA = std::floor((LBF_DATA)randPermA[i] / anglesGrid.size());
		int radiusIdxB = std::floor((LBF_DATA)randPermB[i] / anglesGrid.size());
		int anglesIdxA = randPermA[i] % anglesGrid.size();
		int anglesIdxB = randPermB[i] % anglesGrid.size();

		radiusPairs[i] = std::make_pair(radiusGrid[radiusIdxA], radiusGrid[radiusIdxB]);
		anglePairs[i] = std::make_pair(anglesGrid[anglesIdxA], anglesGrid[anglesIdxB]);
	}

	return;
}

//API functions
void DecisionTree::SetDTParams(int _binLength)//(int _maxDepth, const std::vector<int>& _isSplit)
{
/* _isSplit
 *		: vector structure => N00 / N10 N11 / N20 N21 N22 N23 / N30 N31 N32 N33 N34 N35 N36 N37 / ...
 *			where, N(ij)	i is depth of tree, j is index of i-th nodes
 *
 *		: 0 for don't split
 *		: 1 for split 
 */
// 	assert(pRootNode->sampleIdxs.size() == 0);
// 	assert(_isSplit.size() == pow(2, _maxDepth) - 1);	//condition for # of nodes
// 
// 	maxDepth = _maxDepth;
// 	isSplit = _isSplit;

	binLength = _binLength;

	return;
}

void DecisionTree::Train(const std::vector<Data>& vData, const Params& params, int stage, int lmarkIdx)
{
	/* stage, lmarkIdx is index to access specific parts of vData */
	
	//initialize sampleIdx vector of the root node
	assert(vData.size() > 0);
	pRootNode->sampleIdxs.resize(vData.size() * params.augnumber);
	for (int i = 0; i < pRootNode->sampleIdxs.size(); i++)
	{
		pRootNode->sampleIdxs[i] = i;
	}

	//Train and Split nodes
	/* Split Criteria */
	// 1. the number of leaf node (finally) is fixed by SetDTParams(binLength)
	// 2. select the node to split is based on # of samples (more samples, high priority)
	std::vector<Node*> curLeafNodes;
	curLeafNodes.push_back(pRootNode);
	while (curLeafNodes.size() < binLength)
	{
		//select the node having most samples to split
		int numSamples = curLeafNodes[0]->sampleIdxs.size();
		int targetIdx = 0;
		for (int i = 1; i < curLeafNodes.size(); i++)
		{
			if (curLeafNodes[i]->sampleIdxs.size() > numSamples)
			{
				numSamples = curLeafNodes[i]->sampleIdxs.size();
				targetIdx = i;
			}
		}

		//split the selected node
		Node* selectedNode = curLeafNodes[targetIdx];
		SplitNode(selectedNode, vData, params, stage, lmarkIdx);

		//update the current leaf vector
		curLeafNodes.erase(curLeafNodes.begin() + targetIdx);
		curLeafNodes.insert(curLeafNodes.begin() + targetIdx, selectedNode->plChild);
		curLeafNodes.insert(curLeafNodes.begin() + targetIdx + 1, selectedNode->prChild);
	}

	//setting the vector of leafNodes
	vpLeafNodes.clear();
	vpLeafNodes = curLeafNodes;

	return;
}

void DecisionTree::DeriveBinFeature(const cv::Mat& grayImg, const Params& params, const Bbox curBbox, const LBF_POINT curLandmarkPoint, const cv::Mat& curMeanshape2tf, int stage,
	std::vector<char>& binFeature) const
{
	Node* curNode = pRootNode;
	while (!IsLeafNode(curNode))
	{
		Shape pixelA(1), pixelB(1);
		pixelA.data()[0].x = std::cos(curNode->featPair.first.x) * (curNode->featPair.first.y) * params.maxRatioRadius[stage] * curBbox.width;
		pixelA.data()[0].y = std::sin(curNode->featPair.first.x) * (curNode->featPair.first.y) * params.maxRatioRadius[stage] * curBbox.height;
		pixelB.data()[0].x = std::cos(curNode->featPair.second.x) * (curNode->featPair.second.y) * params.maxRatioRadius[stage] * curBbox.width;
		pixelB.data()[0].y = std::sin(curNode->featPair.second.x) * (curNode->featPair.second.y) * params.maxRatioRadius[stage] * curBbox.height;

		cv::Mat matA, matB;
		pixelA.ToTransMat(matA);
		pixelB.ToTransMat(matB);

		LBF::DoGeometricTransform(curMeanshape2tf, matA, matA);
		LBF::DoGeometricTransform(curMeanshape2tf, matB, matB);

		pixelA.FromTransMat(matA);
		pixelB.FromTransMat(matB);

		pixelA.data()[0].x = std::max(0, std::min(grayImg.cols - 1, (int)(pixelA.data()[0].x + curLandmarkPoint.x + 0.5)));
		pixelA.data()[0].y = std::max(0, std::min(grayImg.rows - 1, (int)(pixelA.data()[0].y + curLandmarkPoint.y + 0.5)));
		pixelB.data()[0].x = std::max(0, std::min(grayImg.cols - 1, (int)(pixelB.data()[0].x + curLandmarkPoint.x + 0.5)));
		pixelB.data()[0].y = std::max(0, std::min(grayImg.rows - 1, (int)(pixelB.data()[0].y + curLandmarkPoint.y + 0.5)));

		int pixelDiff = (int)grayImg.at<uchar>((int)pixelA.data()[0].y, (int)pixelA.data()[0].x) -
						(int)grayImg.at<uchar>((int)pixelB.data()[0].y, (int)pixelB.data()[0].x);

		if (pixelDiff < curNode->threshold)
		{
			curNode = curNode->plChild;
		}
		else
		{
			curNode = curNode->prChild;
		}
	}

	binFeature.clear();
	binFeature.resize(binLength, 0);
	for (int i = 0; i < binLength; i++)
	{
		if (vpLeafNodes[i] == curNode)
		{
			binFeature[i] = 1;
			break;
		}
	}
	return;
}

void DecisionTree::SaveTree(std::ofstream& outStream)
{
	Saver(pRootNode, outStream);

	return;
}

void DecisionTree::LoadTree(std::ifstream& inStream)
{
	std::string rootNodeChecker;
	inStream >> rootNodeChecker;
	assert(/*rootNodeChecker != "X"*/rootNodeChecker == "O");	//tree data should be non-null

	Clear();
	Loader(pRootNode, inStream);

	vpLeafNodes.clear();
	FindLeafNodes(pRootNode);

	return;
}
