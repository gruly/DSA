#!/usr/bin/perl

use utf8;
use Getopt::Long;
use File::Basename;




$home ="/lium/buster1/ghannay/deepSpeech2/deepspeech.pytorch/data";

$files_dir= $ARGV[0];
# eval sclit 

$sclit="/lium/raid01_b/ghannay/phd/tasks/annotation/TOOL/sclite";
$sclit_out= "$files_dir/sclit_out";
system("mkdir -p $sclit_out");
$ref=`ls $files_dir/*reference*`;
$hyp=`ls $files_dir/*transcription*`;
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
