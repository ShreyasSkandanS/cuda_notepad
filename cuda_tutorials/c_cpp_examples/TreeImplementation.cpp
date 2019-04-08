// TreeImplementation.cpp by Shreyas Skandan for ArrayFire started on 2/21/2017

#include "stdafx.h"
#include <iostream>

using namespace std;

struct node_t
{
	// Assuming a binary tree structure with item datatype integer
	int value;
	node_t *left;
	node_t *right;
};

class Tree
{
	private:
		node_t *root;
	
	public:
		Tree();
		~Tree();
		void preorder_traversal(node_t *root);
		node_t *search_element(int key);
		void insert_element(node_t *node_new);
		int find_height(node_t *root);

	private:
		void empty_tree(node_t *root);
};

Tree::Tree()
{
	// Constructor
	root = NULL;
	return;
}

Tree::~Tree()
{
	// Destructor
	empty_tree(root);
	return;
}

void Tree::empty_tree(node_t *root)
{
	// Private function to delete all tree elements
	if (root == NULL)
		return;
	else if (root->left != NULL)
		empty_tree(root->left);
	else if (root->right != NULL)
		empty_tree(root->right);
	else
		delete root;
	return;
}

void Tree::preorder_traversal(node_t *root)
{
	// The following function recursively traverses a binary tree, taking the root node as it's input
	// and displays the elements (values) of the tree separated by new line characters.
	if (root != NULL)
	{
		// First display the element at current node
		cout <<" "<<root->value; 
		cout <<"\n";
		// Traverse the left sub-tree next
		preorder_traversal(root->left);
		// Then proceed to traverse the right subtree
		preorder_traversal(root->right);
	}
}

node_t *Tree::search_element(int key)
{
	// Search for a key element in the tree
	int value = false;
	node_t *temp;
	node_t *clone;

	temp = root;
	while ((temp != NULL) && (temp->value != key))
	{
		if (key < temp->value)
			temp = temp->left;
		else
			temp = temp->right;
	}
	if (temp == NULL)
		return temp;
	else
	{
		// Return a pointer to a clone of the node element
		// instead of returning a pointer to a node in the 
		// tree structure
		clone = new node_t();
		*clone = *temp;
		clone->left = NULL;
		clone->right = NULL;
		return clone;
	}
}

void Tree::insert_element(node_t *node_new)
{
	// Insert a new node into the existing tree structure
	node_t *temp;
	temp = root;
	node_t *prev;
	prev = NULL;

	// Get to the end of the tree (to the leaf layer)
	while (temp != NULL)
	{
		prev = temp;
		if (node_new->value < temp->value)
			temp = temp->left;
		else
			temp = temp->right;
	}
	if (prev == NULL)
	{
		// If the tree is empty
		root = node_new;
		cout << " - Adding " << node_new->value << " to the tree! \n";
	}
	else
	{
		cout << " - Adding " << node_new->value << " to the tree! \n";
		if (node_new->value < prev->value)
			prev->left = node_new;
		else
			prev->right = node_new;
	}
}

int Tree::find_height(node_t *root)
{
	// Finding the maximum height of a given tree structure
	if (root == NULL)
		return 0;
	else
	{
		int lpath_depth = find_height(root->left);
		int rpath_depth = find_height(root -> right);

		// Assuming that we're finding maximum height/depth
		if (lpath_depth > rpath_depth)
			return(lpath_depth + 1);
		else
			return(rpath_depth + 1);
	}
}

int main()
{
	cout << "---------------------------------------------------------\n";
	cout << "----------- Implementing Tree Functions in C++ ----------\n";
	cout << "---------------------------------------------------------\n";
	Tree *tree_test;
	tree_test = new Tree();
	node_t *add_node;
	node_t *root;
	node_t *search_node;

	int search_val = 5;
	int tree_height;

	cout << "\n-------------- A) Insert Nodes Into Tree ----------------\n\n";
	// Manually create a simple tree
	add_node = new node_t();
	add_node->value = 4;
	add_node->left = NULL;
	add_node->right = NULL;
	root = add_node;
	tree_test->insert_element(add_node);

	add_node = new node_t();
	add_node->value = 6;
	add_node->left = NULL;
	add_node->right = NULL;
	tree_test->insert_element(add_node);

	add_node = new node_t();
	add_node->value = 3;
	add_node->left = NULL;
	add_node->right = NULL;
	tree_test->insert_element(add_node);

	add_node = new node_t();
	add_node->value = 10;
	add_node->left = NULL;
	add_node->right = NULL;
	tree_test->insert_element(add_node);

	add_node = new node_t();
	add_node->value = 23;
	add_node->left = NULL;
	add_node->right = NULL;
	tree_test->insert_element(add_node);

	add_node = new node_t();
	add_node->value = 12;
	add_node->left = NULL;
	add_node->right = NULL;
	tree_test->insert_element(add_node);

	cout << "\n Test tree has been created. \n";

	cout << "\n------------- B) Pre Order Tree Traversal ---------------\n";
	tree_test->preorder_traversal(root);

	cout << "\n-------------- C) Search For Node In Tree ---------------\n";
	search_node = tree_test->search_element(search_val);
	if (search_node == NULL)
		cout << "\n Element not found!\n";
	else
		cout << "\n Element found! Value: " << search_node->value << "\n";

	cout << "\n--------------- D) Find Tree Height/Depth ---------------\n";
	tree_height = tree_test->find_height(root);
	cout << " \n Height of the tree is: " << tree_height << "\n";
	cout << "\n---------------------------------------------------------\n\n";

    return 0;
}

