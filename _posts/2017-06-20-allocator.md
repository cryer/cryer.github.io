---
layout: post
title: C++内存管理
description: C++内存管理

---

### 导入

在C++中，我们申请内存一般使用C语言的malloc方式，或者C++的new方式，这两种都是关键字表达式，是我们开发者正常无法介入修改的。但是new其实是可以拆分的，new操作的内部其实是做了三件事：

- operator new

- 类型转换

- 调用构造

其中第一步operator new是可以重载的，因此我们内存管理，主要就是重载operator new，也就是重载`void *operator new(size_t size)`和对应的` void operator delete(void *p)`函数。

在进入下一步之前，我需要申明一下，这篇博客主要讲解C++内存管理的实现方式，也会提一下比较现代化的内存管理方式，但是重点还是前者，因此不会深入讲解具体的内存管理技术。

### 细节

现在针对new的三个拆分进一步展开细节描述：

- operator new分为全局的和类中的，我们重载一般为了不影响其他类，只会重载类中的operator new。其主要的作用就是申请内存，因此在内部可以使用malloc或者继续使用new申请内存，怎么重载operator new内部又用了new？这其实并不影响，因为内部使用的new是使用类似`new char[N]`这种基于第三方类型的new，目的就是申请堆内存，和类本身没有关系，更不会出现递归调用的情况。甚至申请内存也可以用全局的`::operator new`。

- 类型转换没有什么好说的，就是将operator new申请内存的返回指针转换成具体的调用类的类型，类似`Class *p2 = static_cast<Class*>(p1)`

- 调用构造函数：可以直接通过指向Class的p2指针直接调用构造函数，比如`p2->Class::Class(1)`，但是需要注意g++并不支持这样的语法，g++不支持构造之外其他情况下的对构造函数的直接调用，但是MSVC的cl编译器是没有问题的。g++不支持的话则可以使用第二种方式，`placement new`,关于`placement new`之间没有解释，其实就是`operator new`的一种重载形式，正常operator new是`void *operator new(size_t size)`这样，但是我们可以增加参数，当我们增加一个指针参数`void *operator new(size_t size，void *p)`的时候，就变成了`placement new`，调用方式是这样`new(p) Class(1)`。需要注意的是，`placement new`本身并没有调用构造函数，而是我们使用`placement new`之后，编译器会根据`placement new`返回的指针自动类型转换，然后调用构造函数，不需要我们操心。实际`placement new`中只需要直接返回参数的指针即可，什么都不用做。

### 实现

```cpp
class Foo{
    public:
        Foo() = default;
      ~Foo()  = default;
       Foo(int a):m_data(a){}
        //重载operator new
        static void *operator new(size_t size){
            void *p = malloc(size);
            return p;
        }
        //重载placement new
        void *operator new(size_t size, void *p){
            return p;
        }
        // 其他重载
        void *operator new(size_t size, int a, int b){
            void *p = malloc(size);
            return p;
        }
        //重载operator delete
        void operator delete(void *p) noexcept{
            free(p);
        }

    private:
    int m_data;
};
```

这是简单的重载operator new的方式，内部只需要申请对应size的内存，然后返回指针即可。而placement new直接返回指针即可，这个传入placement new的指针，就是外部我们自行申请的内存的指针，或者是第一个operator new重载返回的指针。

可以看到我还写了其他重载，既然我们可以添加一个参数指针，也就可以添加更多的参数，比如上面的例子`void *operator new(size_t size, int a, int b)`,又添加了2个整型int参数，同placement new一样，当我们调用new时，使用`new(a,b) Foo(1);`的时候，就会匹配这个重载，我们可以借此做更多的自定义内容。

这只是展示了如何利用operator new重载进行内存管理，还没有涉及到具体的内存管理策略，下面就实现一个简单的内存管理策略：**每次new这个类的时候，不是申请一个类大小的内存，而是一次性申请很多个类大小内存块，然后每次new的时候，不再申请内存，而是直接从freeList中返回未分配的内存小块即可，同样delete的时候也不直接释放，而是重新加入到freeList中。**

```cpp
class Bar {
public:
    Bar() = default;
    ~Bar() = default;
    Bar(int a) : m_data(a) {};

    void* operator new(size_t);
    void  operator delete(void*); 
      
private:
    Bar* next;
    static Bar* freeList;
    static const int chunkSize;
private:
    int m_data;
};
Bar* Bar::freeList = nullptr;
const int Bar::chunkSize = 20;
```

第一次new直接申请20个类大小的块，使用链表管理。

```cpp
void* Bar::operator new(size_t size)
{
  Bar *p;
  if (!freeList) {
      size_t chunk = chunkSize * size;
      freeList = p =
         reinterpret_cast<Bar*>(new char[chunk]);
//将大块内存分成类大小的小块，并且链表连接起来
      for (; p != &freeList[chunkSize-1]; ++p)
          p->next = p+1;
      p->next = 0;
  }
  p = freeList;
  freeList = freeList->next;
  return p;
}
//重新加入链表，而非直接释放
void Bar::operator delete(void *p)
{
  (static_cast<Bar*>(p))->next = freeList;
  freeList = static_cast<Bar*>(p);
}
```

代码很简单，但是已经在特定情况下有不错的效果了，比如Bar类会多次new实例化的情况下，这种内存管理方式就效果不错了。首先，省去了多次申请内存和释放内存的开销，其次，在内存的使用本身上也是节省的，前提是freeList能够用完的情况下。为什么内存也是节省的呢？因为当我们使用new或者malloc申请内存的时候，返回的并不是单单申请的内存本身，还包括一个cookie的，用来记录本次申请的内存大小，否则free的时候我们怎么知道释放多大的内存呢？cookie中除了记录申请内存大小，还包括debug调试信息和填充对齐区域，只不过我们使用malloc的时候返回的指针直接指向的就是申请的内存数据区域，cookie的部分是对开发者透明的。因此一次性申请只有一块cookie，而多次申请，每次申请都会带一个cookie，因此一次性申请在能够用完的情况下更加节省内存。

当然如果每一个类都重载一次，明显是低效的，因此需要单独写一个allocator内存分配器，专门用来处理内存，然后每个类使用这个分配器即可。如果你了解STL标准模板库，对allocator这个概念就应该非常熟悉。

现在一个问题是，如果是在类的内部重载，我们知道类的具体类型，可以很方便定义类的next指针，现在单独写的话，我们不可能知道类的调用类型，如何定义next指针呢？其实new调用的时候，我们是知道调用类的size的，它是作为参数传递给operator new的，既然知道了具体的size，其实定义什么类型的next都是无所谓的，只要尺寸对的上，加一个类型转换就解决了，因此直接随便定义一个结构体next即可。

具体实现：

```cpp
class allocator 
{
private:
  	struct obj {
    	struct obj* next; 
  	};	
public:
    void* allocate(size_t);
    void  deallocate(void*, size_t);  
private: 
    obj* freeList = nullptr;
    const int CHUNK = 20; 
};

void* allocator::allocate(size_t size)
{
  	obj* p;

  	if (!freeList) {
      	size_t chunk = CHUNK * size;
      	freeList = p = (obj*)malloc(chunk);  
        // 串联所有块   
      	for (int i=0; i < (CHUNK-1); ++i)	{  
           	p->next = (obj*)((char*)p + size);
           	p = p->next;
      	}
      	p->next = nullptr;        
  	}
  	p = freeList;
  	freeList = freeList->next;
 
  	return p;
}

void allocator::deallocate(void* p, size_t)
{
  	((obj*)p)->next = freeList;
  	freeList = (obj*)p;
}
```

注意点：

- 使用自定义结构体的next指针

- 串联所有块的时候直接加上具体的size，而不能采用`p->next = p + 1`这种写法，因为p是自定义的结构体类型指针，跟具体调用的类的类型无关，p+1显然是无意义的。

需要使用内存分配器的类只需要添加static的内存分配器，然后在重载函数中调用分配器的`alocate`和`deallocate`方法，然后在类外初始化内存分配器即可，因为是静态的。

```cpp
class Foo {
public: 
	long L;
	string str;
	static allocator myAlloc;
public:
	Foo(long l) : L(l) {  }
	static void* operator new(size_t size)
  	{     return myAlloc.allocate(size);  	}
  	static void  operator delete(void* pdead, size_t size)
    {     return myAlloc.deallocate(pdead, size);  }
};
allocator Foo::myAlloc;
```

如果想更进一步，还可以把这部分代码定义成宏，写起来更加方便，比如：

```cpp
#define DECLARE_ALLOCATOR()\
public:\
	static void* operator new(size_t size)\
  	{     return myAlloc.allocate(size);  	}\
  	static void  operator delete(void* pdead, size_t size)\
    {     return myAlloc.deallocate(pdead, size);  }\
protected:\
	static allocator myAlloc;

#define IMPLEMENT_ALLOC(class_name)\
	allocator class_name::myAlloc;
```

然后在需要的类中使用：

```cpp
class Foo {
DECLARE_ALLOCATOR()
public: 
	long L;
	string str;
public:
	Foo(long l) : L(l) {  }

};
IMPLEMENT_ALLOC(Foo)
```

### 总结

简单说明了C++内存管理的实现方式，内存分配器的实现方式，以及简单的单链表管理的内存分配策略，实际上，在现代的内存分配策略中，一般都是用多条链表管理，主链表定位不同大小的内存块，从小到大，然后每一个size的位置延伸出这个size大小的内存块组成的freeList链表，可以根据不同的实际尺寸，从内存分配器中选择合适大小的内存块的freeList，然后从这个size的freeList中选出真正的空内存返回给调用方。
