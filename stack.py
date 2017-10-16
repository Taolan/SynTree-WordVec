# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:02:02 2017

@author: Administrator
"""

class Stack(object):
    def __init__(self, size = 8):
        self.stack = []
        self.size = size
        self.top = -1

    def set_size(self, size):
        if self.top >= size:
            raise Exception("StackWillOverFlow")
        self.size = size
    
    def isFull(self):
        return True if self.size == self.top + 1 else False

    def isEmpty(self):
        return True if self.top == -1 else False

    def push(self, data):
        if self.isFull():
            raise Exception("StackOverFlow")
            return
        self.stack.append(data)
        self.top += 1

    def pop(self):
        if self.isEmpty():
            raise Exception("StackIsEmpty")
            return
        self.top -= 1
        return self.stack.pop()
    
    def Top(self):
        if self.isEmpty():
            raise Exception("StackIsEmpty")
            return -1
        return self.stack[self.top]

    def show(self):
        print(self.stack)