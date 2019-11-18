package com.xiaoshuyui.testEquals;

public class Main {

    public static void main(String[] args) {

        MyObject myObject1  = new MyObject(111,"aaa");
        MyObject myObject2  = new MyObject(1,"aaa");



        System.out.println(myObject1.equals((myObject2)));

        System.out.println(myObject1.hashCode());
        System.out.println(myObject2.hashCode());


    }
}
