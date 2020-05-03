class Node(object):
	def __init__(self, value):
		self.value = value
		self.next = None

# example LL
head = Node(2)
head.next = Node(1)

print(head.value)
print(head.next.value)
print(head.next.next)

class LinkedList:
	def __init__(self):
		self.head = None

	def append(self, value):
		if self.head is None:
			self.head = Node(value)
			return

		current_node = self.head
		while current_node.next:
			current_node = current_node.next
		current_node.next = Node(value)
		return

ll =LinkedList()
ll.append(1)
ll.append(2)

node = ll.head
while node:
	print(node.value)
	node = node.next


# Add a method to_list() to LinkedList that 
# converts a linked list back into a Python list.
class LinkedList:
	def __init__(self):
		self.head = None

	def append(self, value):
		if self.head is None:
			self.head = Node(value)
			return

		current_node = self.head
		while current_node.next:
			current_node = current_node.next
		current_node.next = Node(value)
		return

	def to_list(self):
		node_values = []
		node = self.head
		while node:
			node_values.append(node.value)
			node = node.next
		return node_values

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.append(4)
ll.append(5)
print("ll.to_list()",ll.to_list())
print("Pass" if (ll.to_list() == [1,2,3,4,5]) else "Fail")


# DOUBLY LINKEDLIST
print("Doubly LinkedList")

class DoubleNode:
	def __init__(self, value):
		self.value = value 
		self.prev = None
		self.next = None

# Implement a doubly linked list that can 
# append to the tail in constant time.

class DoublyLinkedList:
	def __init__(self):
		self.head = None
		self.tail = None

	def append(self, value):
		if self.head is None:
			self.head = DoubleNode(value)
			self.tail = self.head

		else:
			self.tail.next = DoubleNode(value)
			self.tail.next.prev = self.tail
			self.tail = self.tail.next
		return

linked_list = DoublyLinkedList()
linked_list.append(1.2)
linked_list.append(2.2)
linked_list.append(5.2)
linked_list.append(8.2)

print("Forward through doubly linked list")
current_node = linked_list.head
while current_node:
	print(current_node.value)
	current_node = current_node.next

print("Going backward through doubly linked list ")
current_node = linked_list.tail
while current_node:
	print(current_node.value)
	current_node = current_node.prev

class DoublyLinkedList:
	def __init__(self):
		self.head = None
		self.tail = None

	def append(self, value):
		if self.head is None:
			self.head = DoubleNode(value)
			self.tail = self.head
			return

		self.tail.next = DoubleNode(value)
		self.tail.next.prev = self.tail
		self.tail = self.tail.next
		return

linked_list = DoublyLinkedList()
linked_list.append(1)
linked_list.append(2)
linked_list.append(5)
linked_list.append(8)

print("Forward through doubly linked list")
current_node = linked_list.head
while current_node:
	print(current_node.value)
	current_node = current_node.next

print("Going backward through doubly linked list ")
current_node = linked_list.tail
while current_node:
	print(current_node.value)
	current_node = current_node.prev














