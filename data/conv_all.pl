#!/usr/bin/perl

use utf8

$home = "/lium/buster1/ghannay/deepSpeech2/deepspeech.pytorch/data";
$data_dir="/lium/raid01_b/ghannay/post-doc/data/FR/NE/data_ne";

@l=('dev'); 
#,'dev','test');
for ($i=0; $i <@l; $i++){

	$file_dir=$data_dir."/".$l[$i]."/converted/txt";
	$res_dir=$data_dir."/".$l[$i]."/converted/txt_ne1";
	$cmd=`mkdir -p $res_dir`;
	system($cmd);
	opendir(d,$file_dir);
	my @entrees = readdir(d); 
	print scalar (@entrees),"\n";
	foreach(@entrees){
        	$file = $_;
		if($file=~/.*\.txt/){
#			print $file ,"\n";
			`perl $home/convert_to_ne_only.pl   $file_dir/$file  $res_dir/$file`;
			#system($cmd);
	

		}
	}
}








