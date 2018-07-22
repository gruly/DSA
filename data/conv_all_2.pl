#!/usr/bin/perl


use utf8;
use Encode qw(decode_utf8);
binmode STDOUT, ":utf8";

use Getopt::Long;
use File::Basename;

my $_VERBOSE=0;
my $_HELP;


my %_CONFIG =(
'verbose' => \$_VERBOSE,
'help' => \$_HELP
);

my $usage = "  \
--verbose|--v       : verbose \
--help|h            :   \n";

######################## Parsing parameters with GetOptions
$_HELP = 1 unless GetOptions(\%_CONFIG,'verbose|v', 'help|h');
die "Not enough arguments\n$usage" unless(scalar @ARGV == 1);
#######################################################################"
my $in = $ARGV[0];

open(F1,$in) or die "Can't open file1 \n"; #  ctm
my @l = <F1>;
close(F1);

$home = "/lium/buster1/ghannay/deepSpeech2/deepspeech.pytorch/data";
$data_dir="/lium/raid01_b/ghannay/post-doc/data/FR/NE/data_ne";

$res_dir=$data_dir."/train/converted_aug/txt_ne_CL1";
$cmd=`mkdir -p $res_dir`;

system($cmd);

for ($i=0; $i <@l; $i++){

	$file_dir=$data_dir."/train/converted_aug/txt";
	chomp ($l[$i]);
	$file=$l[$i];
#	print $file_dir,"/",$file ,"\n";
	

	`perl $home/convert_to_ne_only_cl.pl   $file_dir/$file  $res_dir/$file`;

}








