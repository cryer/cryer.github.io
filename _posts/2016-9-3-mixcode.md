---
layout: post
title: C语言和NASM汇编混合编程
description: C语言和NASM汇编混合编程

---

### 导入

今天要谈的不是C语言中用`asm`关键字嵌入汇编，这种一是少见，尤其在现代编译器智能的优化下，更加少见。二是也只是和少量汇编代码的嵌入，今天要谈的是C编译单元和NASM汇编编译单元之间的混合链接。

这看起来很简单，其实还是有一些需要注意的地方的，尤其是汇编调用C编写的函数，很容易出问题，为什么呢？因为调用函数其实涉及到函数参数的压栈以及函数栈的开辟和清理。而高级语言下，编译器会根据调用约定（比如__stdcall,_cdecl）自动处理这些，让上层的开发者无需知道内部发生了什么，但是自己实现时，也就是汇编作为调用者时，这些工作就必须要自己完成。反之C语言调用汇编的函数就简单很多，因为C语言是高级语言，而C语言是调用者，这些你就无需关系。

而32位程序和64位程序又进一步有一些区别。

## 32位

#### c语言调用汇编函数

windows下c语言调用汇编函数`asm_add.asm`：

```nasm
section .text
global _asm_add

_asm_add:
    push ebp
    mov ebp,esp
    mov eax,[ebp+8] ;第一个参数
    add eax, [ebp+12] ;第二个参数
    pop ebp
    ret
```

一个简单的加法函数，几个注意点：

- 32位windows下NASM汇编的函数前需要多加一个下划线，C语言调用的时候去掉下划线

- 参数的位置是这样计算的，这只是32位程序下，64位会使用寄存器传前几个参数
  
  ![](https://github.com/cryer/cryer.github.io/raw/master/image/3.jpg)

C语言书函数`main.c`，就简单调用函数即可：

```c
#include <stdio.h>
extern int asm_add(int a,int b); //声明

int main(){
    int result = asm_add(3, 4);
    printf("result: %d\n", result);
    return 0;
}
```

然后使用：

`nasm -f win32 asm_add.asm`

`gcc -m32 main.c asm_add.o -o demo && demo`

就可以看到输出结果。

如果在Linux下，则去掉汇编函数前的下划线，并且使用`-f elf32`来指明汇编输出格式即可。

### 64位

因为32位程序过于古老，因此不再举例，下面主要用64位进行汇编和C互相调用的举例：

#### C语言调用汇编函数

还是先从简单的C调用汇编开始，一些注意点：

- 32位CPU寄存器不多，所以用栈传参，64位CPU寄存器还是很充足的，所以linux下前6位参数都用寄存器传参，Windows下则是前4个参数，因此linux和windows的参数默认寄存器不同，linux前2个参数是rdi，rsi，但是windows是rcx，rdx

- windows下64位程序，汇编函数名字前也不需要下划线

以Linux举例`add_linux.asm`:

```nasm
section .text
global asm_add   ; 导出符号

; 参数传递: edi (a), esi (b)
; 返回值: eax
asm_add:
    lea eax, [edi + esi]  ; 使用lea实现加法并返回
    ret
```

这里计算加法用了一个小技巧，利用lea指令，lea指令本意是取地址。 [edi+ esi]意思是取 edi+esi的地址内容，也就是把edi和esi寄存器中的数看成地址而不是数字，比如edi是一个int值，比如0x1000，esi也是一个int值，比如0x0040，现在就是要把这两个数看成地址，相加后得到新地址0x1040，然后中括号取地址内容[ ],也就是0x1040地址指向的内容，这个内容具体是什么我们不关心，也不需要知道，甚至可能都没有权限访问，因为我们并不会去访问，而是用lea指令，表示取有效地址，这个命令不会访问具体地址中的数据，0x1040地址指向内容的有效地址自然就是0x1040，所以其实是拿这个我们不关心的地址内容做一个跳板，来间接实现加法，优点是不会改变其他寄存器的值，比如edi和esi。

C文件`main.c`还是一样：

```c
#include <stdio.h>
extern int asm_add(int a,int b); //声明

int main(){
    int result = asm_add(3, 4);
    printf("result: %d\n", result);
    return 0;
}
```

然后使用：

`nasm -f elf64 add_linux.asm`

`gcc  main.c add_linux.o -o demo && ./demo`

就可以看到输出结果。同样，windows下需要把寄存器rdi，rsi换成rcx，rdx

#### 汇编调用c语言函数

汇编调用c语言除了上面说的参数压栈（对64位来说是寄存器），开辟和清理函数栈，还需要注意对齐的问题，x86-64位要求操作数地址必须满足16字节对齐，不仅仅是内存访问效率的优化，也是ABI规范强制要求。

所以要点总结如下：

- 入口点位_start

- 前6个参数通过寄存器（`rdi，rsi，rdx，rcx，r8,r9`）传参（Linux为例）

- 调用函数前保证16字节对齐

汇编`main.asm`:

```nasm
section .text
global _start       ; 汇编入口点
extern c_print      ; 声明C函数

_start:
    mov rbp, rsp    ; 初始化栈基址
    and rsp, -16    ; 16字节对齐

    ; 调用C函数
    mov rdi, 123    ; 第一个参数（整数）
    lea rsi, [msg]  ; 第二个参数（字符串地址）
    call c_print

    ; 退出系统调用
    mov rax, 60     ; sys_exit
    xor rdi, rdi    ; 退出码0
    syscall

section .rodata
msg db "Hello from NASM!", 0
```

主汇编函数，使用了`and rsp,-16`保证16字节对齐，-16转化成16进制就是`FFFF FFFF FFFF FFF0`，最后一位(4bit)是0，其他都是1。`and`取位与之后，rsp的后4bit清零，自然也就16字节对齐。

c文件`c_print.c`:

```c
#include <stdio.h>

void c_print(int num, const char* str) {
    printf("C received: %d and '%s'\n", num, str);
}

```

就一个简单的打印函数，然后进行编译链接：

`nasm -f elf64 main.asm -o main.o`

`gcc -c c_func.c -o c_func.o`

`ld -dynamic-linker /lib64/ld-linux-x86-64.so.2 main.o c_func.o -lc -o demo`

`./demo`

即可看到效果。

而windows下的话，汇编中需要`extern ExitProcess`用来退出程序，并且还要额外预留影子空间，windows前四个参数通过寄存器`rcx,rdx,r8,r9`传递，但调用者还需要栈顶预留32字节影子空间用来保存参数副本，也就是`sub rsp,32`，并且使用link链接器链接的时候，需要使用参数` /LARGEADDRESSAWARE:NO`开启大地址模式，然后用`/entry:main`设定链接的入口函数为main，不过不设置，默认其实是`mainCRTStartup`，内部会进行全局变量初始化，堆内存初始化，标准IO流的设置等，然后才跳转到`main`，但是自己编写的汇编中，不需要你来处理这些，因此要直接设置成入口点`main`,否则会出错。同时注意链接时候的导入库名参数（`kernel32.lib(ExitProcess),ucrt.lib,vcruntime.lib(printf)`）


