package com.xiaoshuyui.kaidian.entity;


import com.xiaoshuyui.kaidian.utils.StatusCode;

public class ExceptionEntity extends BaseEntity {
    @Override
    public Integer getCode() {
        return super.getCode();
    }


    public void setCode(StatusCode statusCode) {
        super.setCode(statusCode.getCode());
    }

    @Override
    public String getMessage() {
        return super.getMessage();
    }

    @Override
    public void setMessage(String message) {
        super.setMessage(message);
    }

    @Override
    public Object getData() {
        return super.getData();
    }

    @Override
    public void setData(String data) {
        super.setData(data);
    }

    public ExceptionEntity(StatusCode code, String message, Object data) {
        super(code, message, data);
    }

    public ExceptionEntity() {
        super();
    }


}
