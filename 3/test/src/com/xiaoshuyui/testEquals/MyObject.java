package com.xiaoshuyui.testEquals;

import java.util.Objects;

public class MyObject extends Object {

    private int id;
    private String name;

    public MyObject(int id, String name) {
        this.id = id;
        this.name = name;
    }

    public MyObject() {
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }



    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof com.xiaoshuyui.testEquals.MyObject)) return false;
        com.xiaoshuyui.testEquals.MyObject myObject = (com.xiaoshuyui.testEquals.MyObject) o;
        return getId() == myObject.getId() &&
                getName().equals(myObject.getName());
    }

    @Override
    public int hashCode() {

        return Objects.hash(getId(), getName());

    }
}
