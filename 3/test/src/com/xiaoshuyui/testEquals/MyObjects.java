package com.xiaoshuyui.testEquals;

import java.util.Objects;

public final class MyObjects  {


    public static int myhash(Object... myObject) {
        return MyArrays.hashCode(myObject);
    }
}
