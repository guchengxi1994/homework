package com.xiaoshuyui.testHash;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class MyThread extends Thread {
    //定义数组
    String s[] = {"zhangsan","张三","lisi","李四","王五","wangwu"};
    private Set s1 ;
    private Set s2 ;

    public Set getS1() {
        return s1;
    }

    public void setS1(Set s1) {
        this.s1 = s1;
    }

    public Set getS2() {
        return s2;
    }

    public void setS2(Set s2) {
        this.s2 = s2;
    }

    public MyThread(String name,Set s1,Set s2) {
        super(name);
        this.s1 = s1;
        this.s2 = s2;
    }

    public void run() {

        for (int i = 0; i < s.length; i++) {
            long hashcode = s[i].hashCode();

            if (this.getName().equals(hashcode%2+"") && "0".equals(this.getName())){
                System.out.println(s[i]+":"+hashcode);
                s1.add(s[i]);
            }

            if (this.getName().equals(hashcode%2+"") && "1".equals(this.getName())){
                System.out.println(s[i]+":"+hashcode);

                s2.add(s[i]);
            }


        }


    }




}
