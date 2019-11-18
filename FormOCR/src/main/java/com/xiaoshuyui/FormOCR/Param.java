package com.xiaoshuyui.FormOCR;

public class Param {
    private String column;
    private String row;
    private String word;
    private String rect;

    public Param(String column, String row, String word,String rect) {
        this.column = column;
        this.row = row;
        this.word = word;
        this.rect = rect;
    }

    public Param() {
    }

    @Override
    public String toString() {
        return "Param{" +
                "column='" + column + '\'' +
                ", row='" + row + '\'' +
                ", word='" + word + '\'' +
                ", rect='" + rect + '\'' +
                '}';
    }

    public String getRect() {
        return rect;
    }

    public void setRect(String rect) {
        this.rect = rect;
    }

    public String getColumn() {
        return column;
    }

    public void setColumn(String column) {
        this.column = column;
    }

    public String getRow() {
        return row;
    }

    public void setRow(String row) {
        this.row = row;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }
}
