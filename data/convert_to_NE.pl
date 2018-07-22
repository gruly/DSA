#!/usr/bin/perl

use utf8;
use Encode qw(decode_utf8);
binmode STDOUT, ":utf8";
use Getopt::Long;
use File::Basename;

$in=$ARGV[0];
$out=$ARGV[1];
open(F1,$in) or die "Can't open $in \n";
@FILE1 = <F1>;
unless (open (OUT, ">$out"))
   {
                 print OUT ("Erreur de creation $out\n");

   }

%map =('[' => 'pers', '(' => 'func', '{' => 'org', '$' => 'loc', '&' => 'prod', '%' => 'amount', '#' => 'time', ')' => 'event' );
# ']' => 'end'


for ($k=0; $k< @FILE1; $k++){
	$ligne=@FILE1[$k];
        chomp($ligne);
        $ligne = decode_utf8($ligne);
	$ligne =~s/\(/ \(/g;
	$ligne =~s/ +/ /g;
	@words = split(/\s+/, $ligne);
	$sent_id=$words[$#words];
	pop ( @words);
	$l2=join(" ",@words);
#	print "-- > ", $l2,"\n";
	$l2 =~s/\*/ \* /g;
	$l2 =~s/\)/ \) /g;
	$l2 =~s/\]/ \] /g;
	$l2 =~s/\[/ \[ /g;
	$l2 =~s/\{/ \{ /g;
	$l2 =~s/\$/ \$ /g;
	$l2 =~s/\&/ \& /g;
	$l2 =~s/\%/ \% /g;
	$l2 =~s/\#/ \# /g;
        $l2 =~s/ +/ /g;
        $l2 =~s/\s*$//g;
        $l2 =~s/^\s*//g;
	@words2 = split(/\s+/, $l2);
	$dne="";
        $value="";
	for($i=0; $i<scalar(@words2); $i++){
		if (  %map{$words2[$i]} ne "" ){
			$dne=$words2[$i];	
		} 
		elsif ($words2[$i] eq ']' and $dne ne "" ){
			$value.=$map{$dne}." ";
			$dne="";
		}
		elsif (%map{$words2[$i]} ne ""  and  $dne ne ""){
			$value.=$map{$dne}." ";
                        $dne="";
		}
        }
	if ($dne ne ""){
		$value.=$map{$dne};
	}
	print OUT $value," ",$sent_id ,"\n";
	


}
