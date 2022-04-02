#!/usr/bin/perl -w
use File::Basename;
use FindBin qw($Bin $Script);
my $binfile="../config/hg19_20k_list";
my $dir=$ARGV[0];
my $sam_gc_readnum_result=$ARGV[1];
my $basename=basename($sam_gc_readnum_result);
my $aftergccorrect="$dir/$basename.gccorrect.filtered";

my %bin20kb;
open IN, $binfile;
while(<IN>){
	chomp;
	my @c=split(/\t/);
	$bin20kb{$c[0]}{$c[1]}{gc}=$c[3];
	$bin20kb{$c[0]}{$c[1]}{ifuse}=$c[4];
}
close IN;


my %chrgclist=();
my %gcdiff=();
my %total_array=();
my @chrarray=('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y');
my @gcbin_median=();
my %chrCorrect=();
my %chr_stat=();
my %chr_zscore=();

my $min_gc=0.20;#bin.gc..
my $max_gc=0.80;#bin.gc..
my $min_total_gc=0.1;#bin.reads....bin.....,.......
my $max_total_gc=10;#bin.reads....bin.....,.......
my $min_bin=0.9;#....bin......bin................
my $max_bin=3;#....bin......bin................

my $min_X_bin=0.7;#X....bin......bin..........................X.....................
my $max_X_bin=3;#X....bin......bin................

print "$sam_gc_readnum_result\n";
open(IN,"$sam_gc_readnum_result")||die "$!";
my $tmpline=<IN>; chomp $tmpline;
my @samsname=split("\t",$tmpline);
while(<IN>){
	chomp;
	next if (/^#/);
	my $str=$_;
	my @array=split(/\t/,$str);
	#next unless($array[9]=~/\d+/);
    my $loc1=$array[1];$loc1=int($loc1/20000);
		next if ($bin20kb{$array[0]}{$array[1]}{gc} eq "NA");
		$gc_group=sprintf("%.3f",$bin20kb{$array[0]}{$array[1]}{gc});
		unless ($array[0] eq "chrX" || $array[0] eq "chrY"){
			next if($gc_group>$max_gc);
	        next if($gc_group<$min_gc);
			for(my $i=3;$i<=$#array;$i++){
				push (@{$total_array{$i}},$array[$i]);
			}
		}
}
close(IN);
my %tBinMedian;
my %min_count;my %max_count;
my @sams;
foreach my $samid(sort{$a<=>$b} keys %total_array){
	push (@sams,$samid);
	my @tmp=@{$total_array{$samid}};
	my $tmpmedian=get_median(@tmp);
	$tBinMedian{$samid}=$tmpmedian;
	$min_count{$samid}=$min_total_gc*$tBinMedian{$samid};
	$max_count{$samid}=$max_total_gc*$tBinMedian{$samid};
#print "$samid\t$tmpmedian\n";
}

my %total_reads_num;
my %total_reads_count;
open(IN,"$sam_gc_readnum_result")||die "$!";
while(<IN>){
    chomp;
	next if (/^#/);
	my @array=split(/\t/);
	next unless($array[9]=~/\d+/);
				next if ($bin20kb{$array[0]}{$array[1]}{gc} eq "NA");
			    my $gc_group=sprintf("%.3f",$bin20kb{$array[0]}{$array[1]}{gc});
				unless ($array[0] eq "chrX" || $array[0] eq "chrY"){
				for(my $i=3;$i<=$#array;$i++){
					next if($gc_group>$max_gc);
				    next if($gc_group<$min_gc);
					next if($array[9]<$min_count{$i});
	                next if($array[9]>$max_count{$i});
					$total_reads_count{$i}++;
					$total_reads_num{$i}+=$array[$i];
					push (@{$gclist{$gc_group}{$i}},$array[$i]);
				}
			}
}
close IN; 

my %gcbin_median;

foreach my $gc(sort {$a<=>$b} keys %gclist){
	foreach my $samid (sort {$a<=>$b} keys %{$gclist{$gc}}){
		my @reads_list=@{$gclist{$gc}{$samid}};
		my $gcbin_median_tmp=get_median(@reads_list);
		$gcbin_median{$gc}{$samid}=$gcbin_median_tmp;
#		print "$gc\t$samid\t$gcbin_median_tmp\n";
	}
}
my %gc_median;
foreach my $samid(@sams){
	$gc_median{$samid}=$total_reads_num{$samid}/$total_reads_count{$samid};
}
my %gc_fact;
foreach my $gc(sort {$a<=>$b} keys %gclist){
	foreach my $samid(@sams){
		my $eachgcbinmedian=$gc_median{$samid};
		if (exists $gcbin_median{$gc}{$samid}){$eachgcbinmedian=$gcbin_median{$gc}{$samid}};
#print "$gc\t$samid\t$eachgcbinmedian\n";
#		print "$samid\t$gc_median{$samid}\n";
#		print "line118\t$gc\t$samid\t$eachgcbinmedian\n";
		$gc_fact{$gc}{$samid}=$eachgcbinmedian-$gc_median{$samid};
	}
}

my $total_reads_before=0;
my $total_reads_after=0;
my %chr_before=();
my %chr_after=();
open(OUTN,"> $aftergccorrect ")||die "$!";
print OUTN "$samsname[3]";
for (my $i=4; $i<=$#samsname;$i++){
	print OUTN "\t$samsname[$i]";
}
print OUTN "\n";
open(IN,"$sam_gc_readnum_result")||die "$!";
while(<IN>){
    chomp;
	next if (/^#/);
    my @array=split(/\t/);
	next if ($bin20kb{$array[0]}{$array[1]}{gc} eq "NA");
	my $gc_group=sprintf("%.3f",$bin20kb{$array[0]}{$array[1]}{gc});
	my $typeuse=$bin20kb{$array[0]}{$array[1]}{ifuse};
	next unless ($typeuse==1);
	foreach my $samid (@sams){
		my $tmpreadnum=$array[$samid]; my $weight=0;
		if (exists $gc_fact{$gc_group}{$samid} && $array[0] ne "chrY"){$weight=$gc_fact{$gc_group}{$samid}};
		$tmpreadnum=$tmpreadnum-$weight; $tmpreadnum=sprintf("%.3f",$tmpreadnum);
		if ($samid==3){
			print OUTN "$tmpreadnum";
		}else{
			print OUTN  "\t$tmpreadnum";
		}
	}
	print  OUTN  "\n";
}

close(OUTN);

#`cat  $aftergccorrect | awk '{if (\$5 ==1) print}' > $outdir/sam_gc_readnum_result_siezeselect.filtered `;

#--------------------------------------------------------------------------------------------------------------
sub get_cv {
	my (@inputs_cnt) = @_;
	my $aver=get_mean(@inputs_cnt);
	my $len=@inputs_cnt;
	my $diff=0;
	foreach my $value(@inputs_cnt){
		$diff=$diff+($aver-$value)**2;
	}
	my $sd_diff=($diff/$len)**0.5;
	my $cv= $sd_diff/$aver;
	return ($cv,$aver,$sd_diff,$len);
}
#--------------------------------------------------------------------------------------------------------------
sub get_mean {
	my (@inputs_cnt) = @_;
	my $sum=0;
	my $len=@inputs_cnt;
	foreach my $str(@inputs_cnt){
		$sum+=$str;
	}
	my $aver=$sum/$len;
	return $aver;
}
#--------------------------------------------------------------------------------------------------------------
sub get_median {
	my (@inputs_cnt) =sort { $a <=> $b } @_;
	my $len=scalar @inputs_cnt;
	if($len%2==1){
		return($inputs_cnt[$len/2]);
	}else{
		my ($upper, $lower);
		$lower=$inputs_cnt[$len/2];
		$upper=$inputs_cnt[$len/2-1];
		return(($lower+$upper)/2);
	}
}
#--------------------------------------------------------------------------------------------------------------
sub get_low_quartle {
	my (@inputs_cnt) =sort { $a <=> $b } @_;
	my $len=scalar @inputs_cnt;
	my $ind=int($len/4);
	return($inputs_cnt[$ind]);
}#--------------------------------------------------------------------------------------------------------------
sub get_up_quartle {
	my (@inputs_cnt) =sort { $a <=> $b } @_;
	my $len=scalar @inputs_cnt;
	my $ind=int($len/4*3);
	return($inputs_cnt[$ind]);
}
