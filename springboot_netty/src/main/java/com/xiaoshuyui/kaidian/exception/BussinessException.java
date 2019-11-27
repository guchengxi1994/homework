package com.xiaoshuyui.kaidian.exception;

import com.xiaoshuyui.kaidian.utils.StatusCode;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class BussinessException extends RuntimeException {

    private StatusCode exceptionCode;

    public BussinessException(StatusCode exceptionCode,String message){
        super(message);
        this.exceptionCode = exceptionCode;
    }

    public BussinessException(StatusCode exceptionCode) {
        this.exceptionCode = exceptionCode;
    }

    public StatusCode getExceptionCode() {
        return exceptionCode;
    }
}
