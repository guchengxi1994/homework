python 实现LRU以及LFU的代码。

LRU是基于collections包下ordereddict类。

LFU是想的用自定义类跟list实现的，不知道对还是错。我觉得是对的。

Extension：

LRU是Least Recently Used的缩写，即最近最少使用，是一种常用的页面置换算法，选择最近最久未使用的页面予以淘汰。

LFU（least frequently used (LFU) page-replacement algorithm）。即最不经常使用页置换算法，要求在页置换时置换引用计数最小的页，因为经常使用的页应该有一个较大的引用次数。


