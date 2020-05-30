use std::{io,cmp::Ordering};
use rand::Rng;

pub fn guess(){
    println!("aaa", );
    let _num = rand::thread_rng().gen_range(1,101);
    let mut gue = String::new();

    println!("{}", _num);
    

    io::stdin().read_line(&mut gue).expect("fail");
    
    println!("{}",gue);

    match gue.cmp(&_num.to_string()) {
        Ordering::Equal =>println!("weee!"),
        Ordering::Less =>println!("fuck!"),
        Ordering::Greater =>println!("lol!"),
    }
}