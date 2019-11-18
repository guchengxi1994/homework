package com.xiaoshuyui.testEquals;

public class MyArrays  {

    public static int hashCode(Object myObject) {
        if (myObject == null)
            return 0;

        int result = 1;

        result = 31*result + myObject.hashCode();

//        for (Object element : myObject)
//            result = 31 * result + (element == null ? 0 : element.hashCode());

        return result;
    }
}
