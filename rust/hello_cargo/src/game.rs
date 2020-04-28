use std::io;

pub fn guess(){
    println!("aaa", );
    let mut gue = String::new();

    io::stdin().read_line(&mut gue).expect("fail");
    
    println!("{}",gue);
}