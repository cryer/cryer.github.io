---
layout: post
title: 一个简易调试器的实现
description: 一个简易调试器的实现

---

### 调试器



利用ptrace实现一个简单的debugger调试器，支持断点，单步调试，continue继续运行，print打印变量。

同时依赖于libdwarf库，dwarf是Linux下调试结构，当我们使用gcc -g编译程序时，生成的.debug节的结构就是dwarf格式的，包含一定调试符号信息，行号信息，行号和源代吗对应信息等。使用dwarfdump命令可以查看带有.debug节的ELF可执行文件的格式详情。



### 完整代码

```c
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/types.h>
#include <fcntl.h>
#include <libdwarf.h>
#include <dwarf.h>

#define MAX_BREAKPOINTS 10
#define MAX_CMD_LEN 256

typedef struct {
    long addr;
    long original_data;
} breakpoint;

breakpoint breakpoints[MAX_BREAKPOINTS];
int bp_count = 0;

long get_load_address(pid_t pid) {
    char maps_path[256];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
    FILE* maps_file = fopen(maps_path, "r");
    if (!maps_file) {
        perror("fopen /proc/pid/maps");
        return 0;
    }

    long load_address = 0;
    char line[512];
    if (fgets(line, sizeof(line), maps_file)) {
        load_address = strtol(line, NULL, 16);
    }

    fclose(maps_file);
    return load_address;
}

void enable_breakpoint(pid_t pid, long addr) {
    if (bp_count >= MAX_BREAKPOINTS) {
        printf("Too many breakpoints.\n");
        return;
    }

    long original_data = ptrace(PTRACE_PEEKDATA, pid, addr, NULL);
    if (original_data == -1) {
        perror("ptrace PEEKDATA");
        return;
    }

    breakpoints[bp_count].addr = addr;
    breakpoints[bp_count].original_data = original_data;
    bp_count++;

    long trap = (original_data & ~0xFF) | 0xCC;
    if (ptrace(PTRACE_POKEDATA, pid, addr, trap) == -1) {
        perror("ptrace POKEDATA");
    }
}

void restore_breakpoint(pid_t pid, int bp_index) {
    if (bp_index < 0 || bp_index >= bp_count) return;

    long addr = breakpoints[bp_index].addr;
    long original_data = breakpoints[bp_index].original_data;

    if (ptrace(PTRACE_POKEDATA, pid, addr, original_data) == -1) {
        perror("ptrace POKEDATA restore");
    }
}

int find_breakpoint(long addr) {
    for (int i = 0; i < bp_count; ++i) {
        if (breakpoints[i].addr == addr) {
            return i;
        }
    }
    return -1;
}

Dwarf_Addr get_line_addr(Dwarf_Debug dbg, const char* file, int line_no, long load_address) {
    Dwarf_Unsigned cu_header_length, abbrev_offset, next_cu_header;
    Dwarf_Half version_stamp, address_size;
    Dwarf_Error error;
    Dwarf_Die cu_die;

    while (dwarf_next_cu_header(dbg, &cu_header_length, &version_stamp, &abbrev_offset, &address_size, &next_cu_header, &error) == DW_DLV_OK) {
        if (dwarf_siblingof(dbg, NULL, &cu_die, &error) != DW_DLV_OK) {
            continue;
        }

        Dwarf_Line *lines;
        Dwarf_Signed line_count;
        if (dwarf_srclines(cu_die, &lines, &line_count, &error) != DW_DLV_OK) {
            dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
            continue;
        }

        for (int i = 0; i < line_count; ++i) {
            Dwarf_Addr line_addr;
            Dwarf_Unsigned line_num;
            char *file_name;

            dwarf_lineaddr(lines[i], &line_addr, &error);
            dwarf_lineno(lines[i], &line_num, &error);
            dwarf_linesrc(lines[i], &file_name, &error);

            if (line_num == line_no && strstr(file_name, file)) {
                 dwarf_dealloc(dbg, lines, DW_DLA_LINE);
                 dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
                 return line_addr + load_address;
            }
             dwarf_dealloc(dbg, file_name, DW_DLA_STRING);
        }
        dwarf_dealloc(dbg, lines, DW_DLA_LINE);
        dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
    }
    return 0;
}

// 获取框架基地址
Dwarf_Addr get_frame_base(Dwarf_Debug dbg, Dwarf_Die cu_die, pid_t pid, Dwarf_Addr pc, Dwarf_Error *error) {
    Dwarf_Attribute fb_attr;
    Dwarf_Die current_die = cu_die;
    struct user_regs_struct regs;

    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1) {
        perror("ptrace GETREGS 失败");
        return 0;
    }

    // 遍历当前 DIE 和所有父 DIE
    while (current_die) {
        Dwarf_Half tag;
        if (dwarf_tag(current_die, &tag, error) == DW_DLV_OK) {
            printf("检查 DIE 标签: 0x%x\n", tag);
        }

        if (dwarf_attr(current_die, DW_AT_frame_base, &fb_attr, error) == DW_DLV_OK) {
            Dwarf_Half form;
            if (dwarf_whatform(fb_attr, &form, error) == DW_DLV_OK) {
                printf("找到 DW_AT_frame_base，格式: %d\n", form);
            }

            Dwarf_Locdesc **llbuf;
            Dwarf_Signed listlen;
            if (dwarf_loclist_n(fb_attr, &llbuf, &listlen, error) == DW_DLV_OK) {
                Dwarf_Addr frame_base = 0;
                for (Dwarf_Signed i = 0; i < listlen; ++i) {
                    Dwarf_Locdesc *loc = llbuf[i];
                    if (loc->ld_cents == 0) continue;

                    // 检查 PC 是否在位置描述的范围内
                    if (loc->ld_lopc == 0 && loc->ld_hipc == 0 || 
                        (pc >= loc->ld_lopc && pc < loc->ld_hipc)) {
                        for (int j = 0; j < loc->ld_cents; ++j) {
                            Dwarf_Small op = loc->ld_s[j].lr_atom;
                            Dwarf_Signed offset = loc->ld_s[j].lr_number;
                            printf("DW_AT_frame_base 表达式: 操作 0x%x, 偏移 %ld\n", op, offset);

                            if (op == DW_OP_regx) {
                                if (offset == 6) frame_base = regs.rbp;
                                else if (offset == 7) frame_base = regs.rsp;
                            } else if (op == DW_OP_fbreg) {
                                frame_base = regs.rbp + offset;
                            } else if (op >= DW_OP_reg0 && op <= DW_OP_reg31) {
                                int reg_num = op - DW_OP_reg0;
                                if (reg_num == 6) frame_base = regs.rbp;
                                else if (reg_num == 7) frame_base = regs.rsp;
                            }
                        }
                        break;
                    }
                }

                for (Dwarf_Signed i = 0; i < listlen; ++i) {
                    dwarf_dealloc(dbg, llbuf[i], DW_DLA_LOCDESC);
                }
                dwarf_dealloc(dbg, llbuf, DW_DLA_LIST);
                dwarf_dealloc(dbg, fb_attr, DW_DLA_ATTR);
                if (frame_base != 0) {
                    printf("找到 DW_AT_frame_base，基地址 = 0x%lx\n", frame_base);
                    return frame_base;
                }
            } else {
                printf("解析 DW_AT_frame_base 失败: %s\n", dwarf_errmsg(error));
            }
            dwarf_dealloc(dbg, fb_attr, DW_DLA_ATTR);
        }

        // 查找父 DIE
        Dwarf_Die parent_die;
        if (dwarf_siblingof(dbg, current_die, &parent_die, error) != DW_DLV_OK) {
            break;
        }
        if (current_die != cu_die) {
            dwarf_dealloc(dbg, current_die, DW_DLA_DIE);
        }
        current_die = parent_die;
    }

    // 回退到 RBP 或 RSP
    printf("警告: 未找到 DW_AT_frame_base，尝试使用 RBP = 0x%lx\n", regs.rbp);
    return regs.rbp;
}


void evaluate_location(Dwarf_Debug dbg, Dwarf_Locdesc *loc, pid_t pid, Dwarf_Addr frame_base, Dwarf_Addr pc, const char* var_name) {
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1) {
        perror("ptrace GETREGS 失败");
        return;
    }

    if (loc->ld_cents == 0) {
        printf("变量 %s 的位置表达式为空\n", var_name);
        return;
    }

    Dwarf_Small op = loc->ld_s[0].lr_atom;
    Dwarf_Signed offset = loc->ld_s[0].lr_number;

    if (op == DW_OP_addr) {
        long var_addr = offset;
        printf("尝试读取地址 0x%lx\n", var_addr);
        errno = 0;
        long var_value = ptrace(PTRACE_PEEKDATA, pid, var_addr, NULL);
        if (var_value != -1 || errno == 0) {
            printf("变量 %s 的值（地址 0x%lx）：%ld (0x%lx)\n", var_name, var_addr, var_value, var_value);
        } else {
            perror("ptrace PEEKDATA 失败");
        }
    } else if (op == DW_OP_fbreg) {
        long var_addr = frame_base + offset;
        printf("尝试读取框架基址 0x%lx + 偏移 %ld = 地址 0x%lx\n", frame_base, offset, var_addr);
        errno = 0;
        long var_value = ptrace(PTRACE_PEEKDATA, pid, var_addr, NULL);
        if (var_value != -1 || errno == 0) {
            int int_value = (int)var_value;
            printf("变量 %s 的值（框架基址 + %ld，地址 0x%lx）：%d (0x%x)\n", 
                   var_name, offset, var_addr, int_value, int_value);
        } else {
            perror("ptrace PEEKDATA 失败");
        }
    } else if (op >= DW_OP_reg0 && op <= DW_OP_reg31) {
        int reg_num = op - DW_OP_reg0;
        long var_value = 0;
        if (reg_num == 0) var_value = regs.rax;
        else if (reg_num == 1) var_value = regs.rdx;
        else if (reg_num == 2) var_value = regs.rcx;
        else if (reg_num == 3) var_value = regs.rbx;
        else if (reg_num == 4) var_value = regs.rsi;
        else if (reg_num == 5) var_value = regs.rdi;
        else if (reg_num == 6) var_value = regs.rbp;
        else if (reg_num == 7) var_value = regs.rsp;
        printf("变量 %s 的值（寄存器 %d）：%ld (0x%lx)\n", var_name, reg_num, var_value, var_value);
    } else if (op == DW_OP_regx) {
        long var_value = 0;
        if (offset == 6) var_value = regs.rbp;
        else if (offset == 7) var_value = regs.rsp;
        printf("变量 %s 的值（扩展寄存器 %ld）：%ld (0x%lx)\n", var_name, offset, var_value, var_value);
    } else {
        printf("变量 %s 的位置操作 0x%x 不支持\n", var_name, op);
    }
}


// 评估 DW_FORM_exprloc 表达式
void evaluate_exprloc(Dwarf_Debug dbg, Dwarf_Attribute attr, pid_t pid, Dwarf_Addr frame_base, const char* var_name) {
    Dwarf_Error error;
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1) {
        perror("ptrace GETREGS 失败");
        return;
    }

    Dwarf_Locdesc *locdesc;
    Dwarf_Signed loccount;
    if (dwarf_loclist(attr, &locdesc, &loccount, &error) != DW_DLV_OK) {
        printf("解析 DW_FORM_exprloc 失败: %s\n", dwarf_errmsg(error));
        return;
    }

    if (loccount == 0) {
        printf("变量 %s 的表达式位置为空\n", var_name);
        dwarf_dealloc(dbg, locdesc, DW_DLA_LOCDESC);
        return;
    }

    // 处理所有位置描述
    for (Dwarf_Signed i = 0; i < loccount; ++i) {
        Dwarf_Locdesc *loc = locdesc + i;
        if (loc->ld_cents == 0) {
            printf("变量 %s 的位置描述 %ld 为空\n", var_name, i);
            continue;
        }

        // 遍历所有操作
        for (int j = 0; j < loc->ld_cents; ++j) {
            Dwarf_Small op = loc->ld_s[j].lr_atom;
            Dwarf_Signed offset = loc->ld_s[j].lr_number;
            printf("解析 DW_FORM_exprloc: 操作 %d/%d, 操作码 0x%x, 偏移 %ld\n", 
                   j + 1, loc->ld_cents, op, offset);

            if (op == DW_OP_addr) {
                long var_addr = offset;
                printf("尝试读取地址 0x%lx\n", var_addr);
                errno = 0;
                long var_value = ptrace(PTRACE_PEEKDATA, pid, var_addr, NULL);
                if (var_value != -1 || errno == 0) {
                    int int_value = (int)var_value;
                    printf("变量 %s 的值（地址 0x%lx）：%d (0x%x)\n", 
                           var_name, var_addr, int_value, int_value);
                } else {
                    perror("ptrace PEEKDATA 失败");
                }
            } else if (op == DW_OP_fbreg) {
                long var_addr = frame_base + offset;
                printf("尝试读取框架基址 0x%lx + 偏移 %ld = 地址 0x%lx\n", 
                       frame_base, offset, var_addr);
                errno = 0;
                long var_value = ptrace(PTRACE_PEEKDATA, pid, var_addr, NULL);
                if (var_value != -1 || errno == 0) {
                    int int_value = (int)var_value;
                    printf("变量 %s 的值（框架基址 + %ld，地址 0x%lx）：%d (0x%x)\n", 
                           var_name, offset, var_addr, int_value, int_value);
                } else {
                    perror("ptrace PEEKDATA 失败");
                }
            } else if (op >= DW_OP_reg0 && op <= DW_OP_reg31) {
                int reg_num = op - DW_OP_reg0;
                long var_value = 0;
                if (reg_num == 0) var_value = regs.rax;
                else if (reg_num == 1) var_value = regs.rdx;
                else if (reg_num == 2) var_value = regs.rcx;
                else if (reg_num == 3) var_value = regs.rbx;
                else if (reg_num == 4) var_value = regs.rsi;
                else if (reg_num == 5) var_value = regs.rdi;
                else if (reg_num == 6) var_value = regs.rbp;
                else if (reg_num == 7) var_value = regs.rsp;
                printf("变量 %s 的值（寄存器 %d）：%ld (0x%lx)\n", 
                       var_name, reg_num, var_value, var_value);
            } else if (op == DW_OP_regx) {
                long var_value = 0;
                if (offset == 6) var_value = regs.rbp;
                else if (offset == 7) var_value = regs.rsp;
                printf("变量 %s 的值（扩展寄存器 %ld）：%ld (0x%lx)\n", 
                       var_name, offset, var_value, var_value);
            } else {
                printf("变量 %s 的表达式位置操作 0x%x 不支持\n", var_name, op);
            }
        }
    }

    dwarf_dealloc(dbg, locdesc, DW_DLA_LOCDESC);
}

// 递归遍历 DIE 树查找变量
void get_die_and_siblings(Dwarf_Debug dbg, Dwarf_Die in_die, int level, const char* var_name, long load_address, pid_t pid, Dwarf_Addr pc) {
    Dwarf_Die cur_die = in_die;
    Dwarf_Error error;
    Dwarf_Die child_die;

    // 检查父 DIE 的作用域
    Dwarf_Addr low_pc, high_pc;
    bool in_scope = true;
    Dwarf_Half tag;
    if (dwarf_tag(cur_die, &tag, &error) == DW_DLV_OK) {
        if (tag == DW_TAG_subprogram || tag == DW_TAG_lexical_block) {
            if (dwarf_lowpc(cur_die, &low_pc, &error) == DW_DLV_OK &&
                dwarf_highpc(cur_die, &high_pc, &error) == DW_DLV_OK) {
                in_scope = (pc >= low_pc + load_address && pc < high_pc + load_address);
                printf("作用域检查: low_pc=0x%lx, high_pc=0x%lx, pc=0x%lx, in_scope=%d\n",
                       low_pc + load_address, high_pc + load_address, pc, in_scope);
            }
        }
    }

    // 处理当前 DIE
    char* die_name = NULL;
    if (dwarf_diename(cur_die, &die_name, &error) == DW_DLV_OK) {
        if (dwarf_tag(cur_die, &tag, &error) == DW_DLV_OK && tag == DW_TAG_variable && strcmp(die_name, var_name) == 0) {
            printf("找到变量 %s 的 DIE\n", var_name);
            Dwarf_Attribute attr;
            if (dwarf_attr(cur_die, DW_AT_location, &attr, &error) == DW_DLV_OK) {
                Dwarf_Half form;
                if (dwarf_whatform(attr, &form, &error) == DW_DLV_OK) {
                    printf("变量 %s 的 DW_AT_location 格式: %d\n", var_name, form);
                    if (form == DW_FORM_exprloc) {
                        Dwarf_Addr frame_base = get_frame_base(dbg, cur_die, pid, pc, &error);
                        evaluate_exprloc(dbg, attr, pid, frame_base, var_name);
                    } else if (form == DW_FORM_block || form == DW_FORM_block1 || form == DW_FORM_data4 || form == DW_FORM_data8) {
                        Dwarf_Locdesc **llbuf;
                        Dwarf_Signed listlen;
                        if (dwarf_loclist_n(attr, &llbuf, &listlen, &error) == DW_DLV_OK) {
                            Dwarf_Addr frame_base = get_frame_base(dbg, cur_die, pid, pc, &error);
                            bool found_valid_loc = false;
                            for (Dwarf_Signed i = 0; i < listlen; ++i) {
                                Dwarf_Locdesc *loc = llbuf[i];
                                if (loc->ld_lopc == 0 && loc->ld_hipc == 0 ||
                                    (pc >= loc->ld_lopc + load_address && pc < loc->ld_hipc + load_address)) {
                                    if (in_scope) {
                                        printf("变量 %s 在作用域内，PC = 0x%lx\n", var_name, pc);
                                        evaluate_location(dbg, loc, pid, frame_base, pc, var_name);
                                        found_valid_loc = true;
                                    }
                                }
                            }
                            if (!found_valid_loc) {
                                printf("变量 %s 的位置不在当前 PC 范围内\n", var_name);
                            }
                            for (Dwarf_Signed i = 0; i < listlen; ++i) {
                                dwarf_dealloc(dbg, llbuf[i], DW_DLA_LOCDESC);
                            }
                            dwarf_dealloc(dbg, llbuf, DW_DLA_LIST);
                        } else {
                            printf("变量 %s 无有效位置列表（dwarf_loclist_n 失败: %s）\n", var_name, dwarf_errmsg(error));
                        }
                    } else {
                        printf("变量 %s 的 DW_AT_location 格式不受支持: %d\n", var_name, form);
                    }
                } else {
                    printf("无法获取变量 %s 的 DW_AT_location 格式: %s\n", var_name, dwarf_errmsg(error));
                }
                dwarf_dealloc(dbg, attr, DW_DLA_ATTR);
            } else {
                printf("变量 %s 无 DW_AT_location 属性: %s\n", var_name, dwarf_errmsg(error));
            }
        }
        dwarf_dealloc(dbg, die_name, DW_DLA_STRING);
    }

    // 递归处理子节点
    if (dwarf_child(cur_die, &child_die, &error) == DW_DLV_OK) {
        get_die_and_siblings(dbg, child_die, level + 1, var_name, load_address, pid, pc);
    }

    // 处理兄弟节点
    Dwarf_Die sib_die;
    if (dwarf_siblingof(dbg, cur_die, &sib_die, &error) == DW_DLV_OK) {
        get_die_and_siblings(dbg, sib_die, level, var_name, load_address, pid, pc);
    }

    dwarf_dealloc(dbg, cur_die, DW_DLA_DIE);
}



void print_variable(Dwarf_Debug dbg, const char* var_name, long load_address, pid_t pid) {
    Dwarf_Unsigned cu_header_length, abbrev_offset, next_cu_header;
    Dwarf_Half version_stamp, address_size;
    Dwarf_Error error;

    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1) {
        perror("ptrace GETREGS 失败");
        return;
    }
    Dwarf_Addr pc = regs.rip;
    printf("当前程序计数器（RIP）= 0x%lx\n", pc);

    bool found = false;
    while (dwarf_next_cu_header(dbg, &cu_header_length, &version_stamp, &abbrev_offset, &address_size, &next_cu_header, &error) == DW_DLV_OK) {
        Dwarf_Die cu_die;
        if (dwarf_siblingof(dbg, NULL, &cu_die, &error) != DW_DLV_OK) {
            printf("无法获取编译单元 DIE: %s\n", dwarf_errmsg(error));
            continue;
        }
        found = true;
        get_die_and_siblings(dbg, cu_die, 0, var_name, load_address, pid, pc);
        dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
    }
    if (!found) {
        printf("未找到任何编译单元或 DWARF 信息无效: %s\n", dwarf_errmsg(error));
    } else {
        printf("完成变量 %s 的搜索\n", var_name);
    }
}

void run_debugger(pid_t child_pid, const char* prog_name) {
    int wait_status;
    long load_address = 0;
    Dwarf_Debug dbg = 0;
    Dwarf_Error err;
    int fd = -1;

    wait(&wait_status);

    load_address = get_load_address(child_pid);
    printf("加载地址: 0x%lx\n", load_address);

    fd = open(prog_name, O_RDONLY);
    if (fd < 0) {
        perror("open 失败");
        return;
    }

    // 检查 ELF 文件头
    char buffer[4];
    if (read(fd, buffer, 4) < 4 || strncmp(buffer, "\177ELF", 4) != 0) {
        fprintf(stderr, "目标文件 %s 不是有效的 ELF 文件\n", prog_name);
        close(fd);
        return;
    }
    lseek(fd, 0, SEEK_SET);

    Dwarf_Handler errhand = 0;
    if (dwarf_init(fd, DW_DLC_READ, errhand, NULL, &dbg, &err) != DW_DLV_OK) {
        fprintf(stderr, "DWARF 初始化失败: %s\n", dwarf_errmsg(err));
        close(fd);
        return;
    }

    char command[MAX_CMD_LEN];
    while (1) {
        printf("(dbg) ");
        fgets(command, MAX_CMD_LEN, stdin);

        if (strncmp(command, "b ", 2) == 0) {
            int line_no = atoi(command + 2);
            char file[100];
            sscanf(command + 2, "%s %d", file, &line_no);
            Dwarf_Addr addr = get_line_addr(dbg, file, line_no, load_address);
            if (addr != 0) {
                printf("设置断点在地址 0x%lx\n", addr);
                enable_breakpoint(child_pid, addr);
            } else {
                printf("无法在 %s:%d 设置断点\n", file, line_no);
            }
        } else if (strncmp(command, "c", 1) == 0) {
            struct user_regs_struct regs;
            if (ptrace(PTRACE_GETREGS, child_pid, NULL, &regs) == -1) {
                perror("ptrace GETREGS 失败");
                continue;
            }
            int bp_index = find_breakpoint(regs.rip - 1);
            if (bp_index != -1) {
                restore_breakpoint(child_pid, bp_index);
                regs.rip -= 1;
                if (ptrace(PTRACE_SETREGS, child_pid, NULL, &regs) == -1) {
                    perror("ptrace SETREGS 失败");
                    continue;
                }
                if (ptrace(PTRACE_SINGLESTEP, child_pid, NULL, NULL) == -1) {
                    perror("ptrace SINGLESTEP 失败");
                    continue;
                }
                wait(&wait_status);
                enable_breakpoint(child_pid, breakpoints[bp_index].addr);
            }

            if (ptrace(PTRACE_CONT, child_pid, NULL, NULL) == -1) {
                perror("ptrace CONT 失败");
                continue;
            }
            wait(&wait_status);

            if (WIFSTOPPED(wait_status)) {
                if (ptrace(PTRACE_GETREGS, child_pid, NULL, &regs) == -1) {
                    perror("ptrace GETREGS 失败");
                    continue;
                }
                printf("子进程停止在 RIP = 0x%llx\n", regs.rip);
            }
        } else if (strncmp(command, "s", 1) == 0) {
            if (ptrace(PTRACE_SINGLESTEP, child_pid, NULL, NULL) == -1) {
                perror("ptrace SINGLESTEP 失败");
                continue;
            }
            wait(&wait_status);
            if (WIFEXITED(wait_status)) {
                printf("子进程已退出\n");
                break;
            }
            struct user_regs_struct regs;
            if (ptrace(PTRACE_GETREGS, child_pid, NULL, &regs) == -1) {
                perror("ptrace GETREGS 失败");
                continue;
            }
            printf("子进程停止在 RIP = 0x%llx\n", regs.rip);
        } else if (strncmp(command, "p ", 2) == 0) {
            char var_name[100];
            sscanf(command + 2, "%s", var_name);
            print_variable(dbg, var_name, load_address, child_pid);
        } else if (strncmp(command, "q", 1) == 0) {
            kill(child_pid, SIGKILL);
            break;
        }
    }

    dwarf_finish(dbg, &err);
    close(fd);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program>\n", argv[0]);
        return 1;
    }

    pid_t child_pid = fork();

    if (child_pid == 0) {
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        execl(argv[1], argv[1], NULL);
    } else if (child_pid > 0) {
        run_debugger(child_pid, argv[1]);
    } else {
        perror("fork");
        return 1;
    }

    return 0;
}

```

**简单测试文件**
```c
#include <stdio.h>

void loop_function(int count) {
    int i = 0;
    int total = 0;
    for (i = 1; i <= count; ++i) {
        total += i;
        printf("i = %d, total = %d\n", i, total);
    }
}

int main() {
    int start_value = 5;
    printf("Debugger test program started.\n");
    loop_function(start_value);
    printf("Debugger test program finished.\n");
    return 0;
}
```

### 编译和运行
编译：`gcc -o debugger debugger.c -ldwarf -lelf` 测试文件`gcc -g -o test_program test.c`
运行：`./debugger test_program`

简单效果展示：
![](https://github.com/cryer/cryer.github.io/raw/master/image/111.jpg)

