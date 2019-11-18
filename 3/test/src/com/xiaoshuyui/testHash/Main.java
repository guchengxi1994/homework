package com.xiaoshuyui.testHash;


import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class Main {


    public static void main(String[] args) {

        Set s1 = new HashSet<String>();
        Set s2 = new HashSet<String>();

        MyThread myThread = new MyThread("0",s1,s2);
        MyThread myThread1 = new MyThread("1",s1,s2);

        myThread.start();
        myThread1.start();

        try{

            myThread.join();
            myThread1.join();

            Iterator iterator = s1.iterator();
            Iterator iterator2 = s2.iterator();

            System.out.println("这里是序列1");
            while (iterator.hasNext()){

                System.out.println(iterator.next());
            }
            System.out.println("这里是序列2");
            while (iterator2.hasNext()){

                System.out.println(iterator2.next());
            }
        }catch (InterruptedException e){

            System.out.println(e);
        }






    }
}
