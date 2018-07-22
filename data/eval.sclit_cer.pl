#!/usr/bin/perl

use utf8;
use Getopt::Long;
use File::Basename;




$home ="/lium/buster1/ghannay/deepSpeech2/deepspeech.pytorch/data";

$files_dir= $ARGV[0];
$output_dir = "conv_ne";
$script_dir = $home;
$cmd = "mkdir -p $files_dir/$output_dir";
#$cmd = "mkdir -p $output_dir";
system($cmd);

$cmd = "cd $files_dir ; ls *.txt";
#$cmd = " ls *.txt";
@files = `$cmd`;



foreach(@files){
	$file = $_;
	chomp($file);
#	$cmd = "cat $files_dir/$file | $script_dir/convert_to_NE.pl > $files_dir/$output_dir/$file";
	$cmd =" perl $script_dir/convert_to_NE.pl  $files_dir/$file  $files_dir/$output_dir/$file";
	system($cmd);
}


# eval sclit 

$sclit="/lium/raid01_b/ghannay/phd/tasks/annotation/TOOL/sclite";
$sclit_out= "$files_dir/$output_dir/sclit_out";
system("mkdir -p $sclit_out");
$ref=`ls $files_dir/$output_dir/*reference*`;
$hyp=`ls $files_dir/$output_dir/*transcription*`;
chomp($ref);
chomp($hyp);
$hyp=~s/\n//g;
$ref=~s/\n//g;


$cmd=" $sclit -r $ref -h $hyp  -O $sclit_out  -o dtl pra sum -i spu_id > /dev/null ";
system($cmd);


## extract CER ##

$sys=`ls $sclit_out/*.sys`; 
chomp($sys);
$sys=~s/\n//g;


open( FHdtl, $sys ) || die "couldn't open input $sys\n";

$data_dtl="";
while ( <FHdtl> ) {
    $_=~s/\|//g;
    $_=~s/\+//g;
    $_=~s/\=//g;
    $_=~s/^ *//g;
    $_=~s/  */ /g;
    $data_dtl .= $_;
}
my $cer=999.9;
my @s=();
#print $data_dtl ,"\n";
if($data_dtl =~/.*Sum\/Avg(.*)/){
   @s=split(" ",$1);
   $cer=$s[$#s-1];
}
print $cer;
close(FHdtl);
