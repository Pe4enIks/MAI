#include<iostream>
#include <vector>
#include <algorithm>
#include "my_tree.h"

int pow2(int val)
{
	return (1 << val);
}

IdTree::~IdTree()
{
	delete_node(root);
}

void IdTree::insert(int id)
{
	root = insert(root, id);
}

void IdTree::erase(int id)
{
	root = remove(root, id);
}

TreeNode* IdTree::find(int id)
{
	return find(root, id);
}

void IdTree::delete_node(TreeNode* node) 
{
    if(node == nullptr) { return; }
    delete_node(node->right);
    delete_node(node->left);
    delete node;
}

std::vector<int> IdTree::get_nodes() const 
{
    std::vector<int> result;
    get_nodes(root, result);
    return result;
}

void IdTree::get_nodes(TreeNode* node, std::vector<int>& v) const 
{
    if(node == nullptr) { return; }
    get_nodes(node->left,v);
    v.push_back(node->data);
    get_nodes(node->right, v);
}


TreeNode* IdTree::find(TreeNode* r, int id)
{
	if(r == nullptr || r->data == id) { return r; }
	if(id < r->data) { return find(r->left, id); }
	if(id > r->data) { return find(r->right, id); }
}

int IdTree::height(TreeNode* node)
{
	int h = 0;
	if(node != nullptr)
	{
		int l_height = height(node->left);
		int r_height = height(node->right);
		int max_height = std::max(l_height, r_height);
		h = max_height + 1;
	}
	return h;
}

int IdTree::diff(TreeNode* node)
{
	int l_height = height(node->left);
	int r_height = height(node->right);
	int b_factor = l_height - r_height;
	return b_factor;
}

TreeNode* IdTree::rr_rotation(TreeNode* parent)
{
	TreeNode* node;
	node = parent->right;
	parent->right = node->left;
	node->left = parent;
	return node;
}

TreeNode* IdTree::ll_rotation(TreeNode* parent)
{
	TreeNode* node;
	node = parent->left;
	parent->left = node->right;
	node->right = parent;
	return node;
}

TreeNode* IdTree::lr_rotation(TreeNode* parent)
{
	TreeNode* node;
	node = parent->left;
	parent->left = rr_rotation(node);
	return ll_rotation(parent);
}

TreeNode* IdTree::rl_rotation(TreeNode* parent)
{
	TreeNode* node;
	node = parent->right;
	parent->right = ll_rotation(node);
	return rr_rotation(parent);
}

TreeNode* IdTree::balance(TreeNode* node)
{
	int bal_factor = diff(node);
	if(bal_factor > 1)
	{
		if(diff(node->left) > 0) { node = ll_rotation(node); }
		else { node = lr_rotation(node); }
	}
	else if(bal_factor < -1)
	{
		if(diff(node->right) > 0) { node = rl_rotation(node); }
		else { node = rr_rotation(node); }
	}
	return node;
}

TreeNode* IdTree::find_min(TreeNode* node)
{
	if(node == nullptr) { return nullptr; }
	else if(node->left == nullptr) { return node; }
	else { return find_min(node->left); }
}

TreeNode* IdTree::insert(TreeNode* root, int val)
{
	if(root == nullptr)
	{
		root = new TreeNode;
		root->data = val;
		root->left = nullptr;
		root->right = nullptr;
		return root;
	}
	else if(val < root->data)
	{
		root->left = insert(root->left, val);
		root = balance(root);
	}
	else if(val >= root->data)
	{
		root->right = insert(root->right, val);
		root = balance(root);
	}
	return root;
}

TreeNode* IdTree::remove(TreeNode* root_node, int val)
{
	TreeNode* node;
	if(root_node == nullptr) { return nullptr; }
	else if(val < root_node->data) { root_node->left = remove(root_node->left, val); }
	else if(val >root_node->data) { root_node->right = remove(root_node->right, val); }
	else if(root_node->left && root_node->right) 
	{
		node = find_min(root_node->right);
		root_node->data = node->data;
		root_node->right = remove(root_node->right, root_node->data);
	}
	else 
	{
		node = root_node;
		if(root_node->left == nullptr) { root_node = root_node->right; }
		else if(root_node->right == nullptr) { root_node = root_node->left; }
		delete node;
	}
	if(root_node == nullptr) { return root_node; }
	root_node = balance(root_node);
}
