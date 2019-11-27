package com.xiaoshuyui.kaidian.utils;


public enum  StatusCode    {

    SUCCESS(200, "success", null),

    ERROR(500, "unknown error", "未知异常"),
    DATABASE_ERROR(-1, "unknown error", "数据库异常"),
    NOT_FOUND(404, "api not found", "接口不存在"),
    INVALID_REQUEST(-2, "api not found", "接口不存在"),
    NO_PERMISSION(407, "no permission", "无权限访问");

    StatusCode(Integer code, String message,Object data) {
    }

    private Integer code;
    private String message;
    private Object data;

    public Integer getCode() {
        return code;
    }

    public void setCode(Integer code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Object getData() {
        return data;
    }

    public void setData(Object data) {
        this.data = data;
    }

    @Override
    public String toString() {
        return "StatusCode{" +
                "code=" + code +
                ", message='" + message + '\'' +
                ", data=" + data +
                '}';
    }
}
