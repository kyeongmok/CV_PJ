#ifndef __DECISIONTREE_H
#define __DECISIONTREE_H
#endif

#ifndef __LBFTYPES_H
#include "LBFTypes.h"
#endif

#ifndef __LBFDEF_H
#include "LBFDef.h"
#endif

//openCV library
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

typedef struct _Node
{
	/* tree node information */
	_Node* pParent;
	_Node* plChild;
	_Node* prChild;
	bool isLeaf;
	int depth;

	/* learning data */
	std::vector<int> sampleIdxs;
	LBF_DATA threshold;
	std::pair<LBF_POINT,LBF_POINT> featPair;	// { (angle1, radius1) , (angle2, radius2) }

	//constructor
	_Node()
		: pParent(nullptr), plChild(nullptr), prChild(nullptr)
		, isLeaf(0), depth(0), threshold(0)
	{
	}
} Node;

class DecisionTree
{
public:
	DecisionTree();		//basic constructor
	DecisionTree(const DecisionTree &Other);	//copy constructor
	~DecisionTree();	//destructor

	//API functions
	void SetDTParams(int _binLength);//(int _maxDepth, const std::vector<int>& _isSplit);

	void Train(const std::vector<Data>& vData, const Params& params, int stage, int lmarkIdx);

	void DeriveBinFeature(const cv::Mat& grayImg, const Params& params, const Bbox curBbox, const LBF_POINT curLandmarkPoint, const cv::Mat& curMeanshape2tf, int stage,
		std::vector<char>& binFeature) const;

	void SaveTree(std::ofstream& outStream);
	
	void LoadTree(std::ifstream& inStream);

private:
	Node* pRootNode;
	std::vector<Node*> vpLeafNodes;

	//parameters
	int binLength;

	//private member functions
	inline bool IsLeafNode(Node* pNode) const	{ return !pNode->plChild && !pNode->prChild; }
	void Destroyer(Node* pSubRootNode);
	void FindLeafNodes(Node* pSubRootNode);
	void Saver(Node* pSubRootNode, std::ofstream& outStream);
	void Loader(Node* pSubRootNode, std::ifstream& inStream);
	int GetTreeDepth();
	void Clear();	//sets the DT object to the initial state(after calling basic constructor)
	void SplitNode(Node* pNode, const std::vector<Data>& vData, const Params& params, int stage, int lmarkIdx);
	void GetProposals(int numProposals, const std::vector<LBF_DATA>& radiusGrid, const std::vector<LBF_DATA>& anglesGrid,
		std::vector<std::pair<LBF_DATA, LBF_DATA>>& radiusPairs, std::vector<std::pair<LBF_DATA, LBF_DATA>>& anglePairs);
};